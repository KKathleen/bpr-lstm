# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:18:00 2018

@author: DELL
"""
import data
import torch
import torch.utils.data
import torch.nn.init as init
import bpr_lstm
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
import model_lstm
import midi
import numpy as np

torch.cuda.manual_seed(seed=0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

batch_size = 1024
epoches = 500
no_cuda = False
seed = 1
weight_decay = 0.1
sing_train_path = 'new_data/sing_30_days'
sing_test_path = 'new_data/sing_test_7_days'
listen_path = 'new_data/listen_30_days'
midi_path  = 'new_data/midi.out'


data_loader = data.Data(sing_train_path, sing_test_path, listen_path, batch_size)
midi_data = midi.Midi(midi_path)



# n_factors = 40
model = bpr_lstm.PMF(n_users=data_loader.usr_num, n_items= data_loader.song_num, n_factors=40, no_cuda=no_cuda)

model.cuda()

lstm = model_lstm.MidiLstm(embedding_dim = 100, hidden_dim = 100, pitch_size = 80, nlayers = 2, d = 40)
lstm.cuda()


def bpr_loss(like_i, like_j, sing_i, sing_j):
    loss_listen = (1.0 - F.sigmoid(like_i - like_j))
    loss_sing = (1.0 - F.sigmoid(sing_i - sing_j))
    loss = (loss_listen + loss_sing) / 2.0
    return loss.mean()

def bpr_loss_test(sing_i, sing_j):
    loss = (1.0 - F.sigmoid(sing_i - sing_j))
    return loss.mean()


optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-8)

lr = 20


def train(batch_num, n):
    model.train()
    epoch_loss = 0.0
    lstm.train()
    for i in range(batch_num):
        batch_listen = data_loader.generate_listen_train_batch()
        batch_sing = data_loader.generate_sing_train_batch()
        optimizer.zero_grad()
        lstm.zero_grad()
        uid_listen = Variable(torch.from_numpy(batch_listen[:, 0]).cuda(),  volatile= False)
        sid_i_listen = Variable(torch.from_numpy(batch_listen[:, 1]).cuda(),  volatile= False)
        sid_j_listen = Variable(torch.from_numpy(batch_listen[:, 2]).cuda(),  volatile= False)
        like_i = model.forward_listen(uid_listen, sid_i_listen)
        like_j = model.forward_listen(uid_listen, sid_j_listen)
        
        uid_sing = Variable(torch.from_numpy(batch_sing[:, 0]).cuda(),  volatile= False)
        sid_i_sing = Variable(torch.from_numpy(batch_sing[:, 1]).cuda(),  volatile= False)
        sid_j_sing = Variable(torch.from_numpy(batch_sing[:, 2]).cuda(),  volatile= False)

        midi_i = midi_data.get_batch(sid_i_sing.data.cpu().numpy())
        midi_j = midi_data.get_batch(sid_j_sing.data.cpu().numpy())
        midi_seq = Variable(torch.from_numpy(np.array(midi_i + midi_j)).cuda(), volatile= False)
        init_hidden = lstm.init_hidden(len(sid_i_sing) + len(sid_j_sing))
        d = lstm(init_hidden, midi_seq, len(sid_i_sing) + len(sid_j_sing))
        
        d_i = d[:len(sid_i_sing)]
        d_j = d[len(sid_i_sing):]
        sing_i = model.forward_sing(uid_sing, sid_i_sing, d_i)
        sing_j = model.forward_sing(uid_sing, sid_j_sing, d_j)
        loss = bpr_loss(like_i, like_j, sing_i, sing_j).cuda()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm(lstm.parameters(), 0.1)
        for p in lstm.parameters():
            p.data.add_(-lr, p.grad.data)
        epoch_loss += loss.data[0]
    print("epoch " + str(n) + "\t train auc: " + str(1 - epoch_loss/ batch_num))
    return epoch_loss


def test(test_batch, i):
    lstm.eval()
    uid = Variable(torch.from_numpy(test_batch[:, 0]).cuda(),  volatile= True)
    sid_i = Variable(torch.from_numpy(test_batch[:, 1]).cuda(),  volatile= True)
    sid_j = Variable(torch.from_numpy(test_batch[:, 2]).cuda(),  volatile= True)
    midi_i = midi_data.get_batch(sid_i.data.cpu().numpy())
    midi_j = midi_data.get_batch(sid_j.data.cpu().numpy())
    midi_seq = Variable(torch.from_numpy(np.array(midi_i + midi_j)).cuda(), volatile= True)
    init_hidden = lstm.init_hidden(len(sid_i) + len(sid_j))
    d = lstm(init_hidden, midi_seq, len(sid_i) + len(sid_j))
    d_i = d[:len(sid_i)]
    d_j = d[len(sid_i):]
    sing_i = model.forward_sing(uid, sid_i, d_i)
    sing_j = model.forward_sing(uid, sid_j, d_j)
    loss = bpr_loss_test(sing_i, sing_j)
    print("epoch " + str(i) + "\t test auc: " + str(1 - loss.data[0]))
    return loss.data[0]



test_batch = data_loader.generate_sing_test_batch(n = 5000)
for i in range(50):
    min_loss = 10
    train(300, i)
    cur_loss = test(test_batch, i)
    if cur_loss < min_loss:
        min_loss = cur_loss
    else:
        lr = lr/4
