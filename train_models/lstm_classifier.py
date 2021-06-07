import torch ; import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import datasets
# from transformers import BertTokenizer
import pickle
from absl import app, flags


# Parser
FLAGS = flags.FLAGS

flags.DEFINE_integer("vocab_size", 30522, "")
flags.DEFINE_integer("embedding_dim", 768, "")
flags.DEFINE_integer("hidden_dim", 768, "")
flags.DEFINE_integer("num_classes", 2, "")
flags.DEFINE_integer("batch_size", "32", "")
flags.DEFINE_float("lr", "1e-3", "")
flags.DEFINE_string("train_ds", "/home/sarvasvarora/spark/data/train_ds.pickle", "")
flags.DEFINE_string("test_ds", "/home/sarvasvarora/spark/data/test_ds.pickle", "")
flags.DEFINE_string("save_model", "/home/sarvasvarora/spark/data/lstm_model.pt", "Path to save model")


# Define model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # tokenization is done using BertTokenizer that has a vocabulary size of 30522
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(1, batch_size, self.hidden_dim)
        hidden_b = torch.zeros(1, batch_size, self.hidden_dim)

        if next(self.parameters()).is_cuda:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = torch.autograd.Variable(hidden_a)
        hidden_b = torch.autograd.Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input_ids, x_lengths):
        batch_size, _ = input_ids.size()
        h_0, c_0 = self.init_hidden(batch_size)
        x = self.embedding(input_ids)
        x = self.dropout(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed_x, (h_0, c_0))
        # unpacked_x, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # we don't really need to unpack since we are only using the last hidden state i.e., h_n
        logits = self.fc(h_n)
        logits = torch.squeeze(logits)
        return logits


#Training loop
def train(model, optim, criterion, train_dataloader, epochs=10, save=None):
    device = torch.device('cuda') if next(model.parameters()).is_cuda else torch.device('cpu')
    running_loss = 0.0

    for epoch in range(epochs):
        
        for i, batch in enumerate(train_dataloader):
            model.zero_grad()
            
            input_ids, target = batch['input_ids'].to(device), batch['label'].to(device)
            # input_ids = torch.squeeze(input_ids[torch.nonzero(input_ids)]) # Remove the padding
            input_ids_lengths = [torch.tensor(len(torch.squeeze(torch.nonzero(i)))) for i in input_ids]

            out = model(input_ids, input_ids_lengths)

            loss = criterion(out, target)
            loss.backward()
            optim.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] => Batch = [{i+1}/{len(train_dataloader)}], Loss = {loss.item()}, Running Loss = {running_loss/100}")
                running_loss = 0.0
    
    if save is not None:
        torch.save(model.state_dict(), save)
        

def main(argv):
    # Import data
    with open(FLAGS.train_ds, "rb") as f:
        train_ds = pickle.load(f)

    with open(FLAGS.test_ds, "rb") as f:
        test_ds = pickle.load(f)

    train_dataloader = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=FLAGS.batch_size)

    VOCAB_SIZE = FLAGS.vocab_size
    EMBEDDING_DIM = FLAGS.embedding_dim
    HIDDEN_DIM = FLAGS.hidden_dim
    NUM_CLASSES = FLAGS.num_classes

    model = LSTMClassifier(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[1, 2, 4])
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = torch.nn.CrossEntropyLoss()

    #Train it
    train(model, optim, criterion, train_dataloader, save=FLAGS.save)


if __name__ == '__main__':
    app.run(main)