import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F


class BiLSTM_CRF(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bi_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.lstm_layers, bidirectional=True)

        self.logistic = nn.Linear(2 * self.hidden_dim, self.tag_size)

    def forward(self, sentences):
        sents_ebd = self.lookup_table(sentences)
        output, _ = self.bi_lstm(sents_ebd)
        output = self.logistic(output).view(-1, self.tag_size)
        return F.log_softmax(output)

    def __init_weights(self, scope=0.25):
        self.lookup_table.weight.data.uniform(-scope, scope)
        init.xavier_uniform(self.logistic.weight)

