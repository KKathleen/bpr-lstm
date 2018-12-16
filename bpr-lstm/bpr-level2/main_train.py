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
import numpy as np
import pandas as pd

torch.cuda.manual_seed(seed=0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 1024
epoches = 500
no_cuda = False
seed = 1
weight_decay = 0.1
sing_train_path = '../../new_data/sing_30_days'
sing_test_path = '../../new_data/sing_test_7_days'
listen_path = '../../new_data/listen_30_days'
midi_path  = '../../new_data/midi.out'


data_loader = data.Data(sing_train_path, sing_test_path, listen_path, batch_size)
# n_factors = 40
model = bpr_lstm.PMF(n_users=data_loader.usr_num, n_items= data_loader.song_num, n_factors=30, no_cuda=no_cuda)
model.cuda()



def bpr_loss(like_i, like_j, sing_i, sing_j):
    loss_listen =   -F.logsigmoid(like_i - like_j).sum()
    loss_sing =   -F.logsigmoid(sing_i - sing_j).sum()
    loss = 0.2 * loss_listen + 0.8 * loss_sing
    return loss

def bpr_loss_test(sing_i, sing_j):
    loss =  -F.logsigmoid(sing_i - sing_j).mean()
    return loss

def hinge_loss(like_i, like_j, sing_i, sing_j):
    pass


optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 1e-9)



def train(batch_num, n):
    model.train()
    epoch_loss = 0.0
    for i in range(batch_num):
        batch_listen = data_loader.generate_listen_train_batch()
        batch_sing = data_loader.generate_sing_train_batch()
        optimizer.zero_grad()
        uid_listen = Variable(torch.from_numpy(batch_listen[:, 0]).cuda(), requires_grad = False)
        sid_i_listen = Variable(torch.from_numpy(batch_listen[:, 1]).cuda(),  requires_grad = False)
        sid_j_listen = Variable(torch.from_numpy(batch_listen[:, 2]).cuda(),  requires_grad = False)
        like_i = model.forward_listen(uid_listen, sid_i_listen)
        like_j = model.forward_listen(uid_listen, sid_j_listen)
        uid_sing = Variable(torch.from_numpy(batch_sing[:, 0]).cuda(),  requires_grad = False)
        sid_i_sing = Variable(torch.from_numpy(batch_sing[:, 1]).cuda(),  requires_grad = False)
        sid_j_sing = Variable(torch.from_numpy(batch_sing[:, 2]).cuda(),  requires_grad = False)
        sing_i = model.forward_sing(uid_sing, sid_i_sing)
        sing_j = model.forward_sing(uid_sing, sid_j_sing)
        loss = bpr_loss(like_i, like_j, sing_i, sing_j).cuda()
        loss.backward()
        optimizer.step()
        epoch_loss += bpr_loss_test(sing_i, sing_j) .data[0]
    print("epoch " + str(n) + "\t train auc: " + str(np.exp( - epoch_loss/ batch_num)))
    return epoch_loss


def test_auc(test_batch, i):
    uid = Variable(torch.from_numpy(test_batch[:, 0]).cuda(),  volatile= True)
    sid_i = Variable(torch.from_numpy(test_batch[:, 1]).cuda(),  volatile= True)
    sid_j = Variable(torch.from_numpy(test_batch[:, 2]).cuda(),  volatile= True)
    sing_i = model.forward_sing(uid, sid_i)
    sing_j = model.forward_sing(uid, sid_j)
    loss = bpr_loss_test(sing_i, sing_j)
    print("epoch " + str(i) + "\t test auc: " + str(np.exp(-loss.data[0])))
    return loss.data[0]


def test_map():
    song_num = data_loader.song_num
    result = []
    for uid, sid_set in data_loader.sing_test.items(): 
        sid_num = len(sid_set)
        if sid_num > 0:
            uid = Variable(torch.from_numpy(np.asarray([uid]*song_num)).cuda(),  volatile= True)
            sid = Variable(torch.from_numpy(np.asarray(range(song_num))).cuda(), volatile = True)
            sing_score = model.forward_sing(uid, sid)
            ans = sing_score.data.cpu().numpy()
            ans = -1 * ans
            rank = ans.argsort()
            hit = 0
            map_score = 0
            for loc,r in enumerate(rank):
                if r in sid_set:
                    hit += 1
                    map_score+= hit/(loc+1)
                if hit >= sid_num:
                    break
            result += [map_score/sid_num]
    print("test map", np.mean(result))
    
    
test_batch = data_loader.generate_sing_test_batch_total(neg_num = 1)
for i in range(20):
    min_loss = 10
    train(300, i)
    cur_loss = test_auc(test_batch, i)
    test_map()
