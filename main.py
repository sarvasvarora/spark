from __future__ import division
from __future__ import absolute_import

import os
import sys
import shutil
import re
import time
import random
import argparse
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import datasets
from transformers import BertForSequenceClassification, BertTokenizer
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Linear, quan_LSTM, quantize

from attack.BFA import *
import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np

from models.quan_bert import IMDBSentimentClassifier

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model',
                    default='bert',
                    type=str,
                    help='Model to use',
                    choices=['bert', 'bert_quan'])

parser.add_argument('--train_ds',
                    default='./train_ds.pickle',
                    type=str,
                    help='Path to train dataset')

parser.add_argument('--test_ds',
                    default='./test_ds.pickle',
                    type=str,
                    help='Path to test dataset')

# Logging
parser.add_argument('--save_path',
                    type=str,
                    default='./logs/',
                    help='Folder to save checkpoints and log.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=2,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Evaluation/Validation
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

##########################################################################


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################


def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isfile(args.train_ds):
        raise FileNotFoundError("Train dataloader not found")
    if not os.path.isfile(args.test_ds):
        raise FileNotFoundError("Test dataloader not found")

    with open(args.train_ds, "rb") as f:
        train_ds = pickle.load(f)
        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            drop_last=True,
            shuffle=True)

    with open(args.test_ds, "rb") as f:
        test_ds = pickle.load(f)
        test_loader = DataLoader(
            test_ds,
            batch_size=8,
            drop_last=True,
            shuffle=True)

    print_log("=> creating model {}".format(args.model), log)

    # Init model and criterion
    net = models.__dict__[args.model]()

    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net, args.quan_bitwidth)

    # update the step_size once the model is loaded. This is used for
    # quantization.
    for m in net.modules():
        if isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight
            # init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Linear):
                m.__reset_weight__()
                # print(m.weight)

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)

    if args.enable_bfa:
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                       args.n_iter, log, writer, csv_save_path=args.save_path,
                       random_attack=args.random_bfa)
        return

    if args.evaluate:
        _, _, _, output_summary = validate(
            test_loader, net, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary.csv'),
                                            header=['top-1 output'], index=False)
        return


def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see:
    # https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, batch in enumerate(train_loader):
        data = batch['input_ids']
        target = batch['label']
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    acc, val_loss, output_summary = validate(test_loader, model,
                                             attacker.criterion, log, summary_output=True)
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_BFA_0.csv'),
                  index=False)

    writer.add_scalar('attack/acc', acc, 0)
    writer.add_scalar('attack/val_loss', val_loss, 0)

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    df = pd.DataFrame()  # init a empty dataframe for logging
    last_acc = acc

    for i_iter in range(N_iter):
        print_log('**********************************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)

        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                      log)
            print_log(
                'loss after attack: {:.4f}'.format(
                    attacker.loss_max), log)
        except BaseException:
            pass

        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val/test dataset
        acc, val_loss, output_summary = validate(
            test_loader, model, attacker.criterion, log, summary_output=True)
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = i_iter + 1
        tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_BFA_{}.csv'.format(i_iter + 1)),
                      index=False)

        # add additional info for logging
        acc_drop = last_acc - acc
        last_acc = acc

        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(acc)
            attack_log[i].append(acc_drop)
        # print(attack_log)
        df = df.append(attack_log, ignore_index=True)

        writer.add_scalar('attack/acc', acc, i_iter + 1)
        writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()

        # Stop the attack if the accuracy is below the configured break_acc.
        # if args.dataset == 'cifar10':
        #     break_acc = 11.0
        # elif args.dataset == 'imagenet':
        #     break_acc = 0.2
        # if acc <= break_acc:
        #     break

    # attack profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                   'weight before attack', 'weight after attack', 'validation accuracy',
                   'accuracy drop']
    df.columns = column_list
    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(
            os.path.join(
                csv_save_path,
                csv_file_name),
            index=None)

    return


def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = []  # init a list for output summary

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inp = batch['input_ids']
            target = batch['label']
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                inp = inp.cuda()

            # compute output
            output = model(inp)
            loss = criterion(output, target)

            # summary the output
            if summary_output:
                # get the index of the max log-probability
                tmp_list = output.max(
                    1, keepdim=True)[1].flatten().cpu().numpy()
                output_summary.append(tmp_list)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(loss.item(), inp.size(0))
            acc.update(prec1[0], inp.size(0))

        print_log(
            '  **Test** Prec@1 {acc.avg:.3f} Error@1 {error1:.3f}'
            .format(acc=acc, error1=100 - acc.avg), log)

    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return acc.avg, losses.avg, output_summary
    else:
        return acc.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in [1]:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
