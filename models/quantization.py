import torch
import torch.nn as nn
from torch import _VF
from torch.autograd import Variable
import torch.nn.functional as F


class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  # in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight
        # quantization
        self.inf_with_weight = True


class quan_LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(quan_LSTM, self).__init__(*args, **kwargs)

        self.weight = torch.Tensor().requires_grad_()
        self.__update__()

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  # in-place reverse

    def forward(self, input, hx=None):
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to
        # compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(
                0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        if self.inf_with_weight:
            _f_weights = []
            for i in range(0, len(self._flat_weights), 4):
                _f_weights.append(self._flat_weights[i] * self.step_size)
                _f_weights.append(self._flat_weights[i + 1] * self.step_size)
                _f_weights.append(self._flat_weights[i + 2])
                _f_weights.append(self._flat_weights[i + 3])
        else:
            self.__reset_stepsize__()
            _f_weights = []
            for i in range(0, len(self._flat_weights), 4):
                _f_weights.append(
                    quantize(
                        self._flat_weights[i],
                        self.step_size,
                        self.half_lvls) *
                    self.step_size)
                _f_weights.append(quantize(
                    self._flat_weights[i + 1], self.step_size, self.half_lvls) * self.step_size)
                _f_weights.append(self._flat_weights[i + 2])
                _f_weights.append(self._flat_weights[i + 3])

        if batch_sizes is None:
            result = _VF.lstm(input, hx, _f_weights, self.bias, self.num_layers,
                                 self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, _f_weights, self.bias,
                                 self.num_layers, self.dropout, self.training, self.bidirectional)

        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to
        # compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_packed = nn.utils.rnn.PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight
        # quantization
        self.inf_with_weight = True

    def __update__(self, mode="weight"):
        # update self.weight (and self.weight.grad) with the weights (and their grads) of LSTM
        if mode == "weight":
            weights = []
            grads = []

            for i in range(0, len(self._flat_weights), 4):
                weights.append(self._flat_weights[i].detach().clone())
                weights.append(self._flat_weights[i+1].detach().clone())
                try:
                    grads.append(self._flat_weights[i].grad.detach().clone())
                    grads.append(self._flat_weights[i+1].grad.detach().clone())
                except AttributeError:
                    pass

            self.weight = torch.cat(weights)
            if len(grads) == len(weights):
                self.weight.grad = torch.cat(grads) 

        elif mode == "flat_weights":
            n = len(self._flat_weights)
            flag = True
            weights = torch.split(self.weight, n)
            try:
                grads = torch.split(self.weight.grad, n)
            except AttributeError:
                flag = False

            with torch.no_grad():
                for i in range(0, n, 4):
                    # self._flat_weights[i] = weights[i//2].detach().clone()
                    # self._flat_weights[i+1] = weights[i//2 + 1].detach().clone()
                    self._flat_weights[i].data = weights[i//2].data
                    self._flat_weights[i+1].data = weights[i//2 + 1].data
                    if flag:
                        # self._flat_weights[i].grad = grads[i//2].detach().clone()
                        # self._flat_weights[i+1].grad = grads[i//2 + 1].detach().clone()
                        self._flat_weights[i].grad.data = grads[i//2].data
                        self._flat_weights[i+1].grad.data = grads[i//2 + 1].data

        else:
            return