import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class MidiLstm(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, pitch_size, nlayers, d = 30):
        super(MidiLstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.pitch_embeddings = nn.Embedding(pitch_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = nlayers)
        self.hidden2d = nn.Linear(hidden_dim, d)
        self.encode_dict = {}

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.nlayers, batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(self.nlayers, batch_size, self.hidden_dim).cuda())
        return (h0, c0)

    def forward(self,init_hidden, pitch_seq, batch_size):
        embeds = self.pitch_embeddings(pitch_seq)
        x = embeds.view(100, batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, init_hidden)
        d  = self.hidden2d(lstm_out[-1])
        return d
