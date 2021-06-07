import torch
from torch import nn


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

    def forward(self, input_ids):
        batch_size, _ = input_ids.size()
        x_lengths = [torch.tensor(len(torch.squeeze(torch.nonzero(i)))) for i in input_ids]

        h_0, c_0 = self.init_hidden(batch_size)
        x = self.embedding(input_ids)
        x = self.dropout(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed_x, (h_0, c_0))
        # unpacked_x, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # we don't really need to unpack since we are only using the last hidden state i.e., h_n
        logits = self.fc(h_n)
        logits = torch.squeeze(logits)
        return logits


def lstm():
    model = LSTMClassifier(30522, 768, 768, 2)
    model.load_state_dict(torch.load("/home/sarvasvarora/spark/data/lstm_model.pt", map_location=torch.device('cpu')))
    return model