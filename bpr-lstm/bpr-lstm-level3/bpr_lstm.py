# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:27 2018

@author: DELL
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from numpy.random import RandomState


class PMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=30, is_sparse=False, no_cuda=False):
        super(PMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.no_cuda = no_cuda
        self.random_state = RandomState(1)
        # U: user's latent features for interests
        self.U = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.U.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()
        # V: song's latent features for interests
        self.V = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.V.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()
        # C: user's latent features for performance 
        self.C = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.C.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()
        # R: user's latent features for feedbacks of performance
        self.R = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.R.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()
        # D: song's latend features for performance
        self.D = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.D.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()
    
    def forward_listen(self, users_index, items_index):
        u = self.U(users_index)
        v = self.V(items_index)
        like = (u * v).sum(1)
        return like

    def forward_sing(self, users_index, items_index):
        u = self.U(users_index)
        v = self.V(items_index)
        c = self.C(users_index)
        d = self.D(items_index)
        sing = (u * v).sum(1) + (c * d).sum(1)
        return sing
    
    def forward_feedback(self, users_index, items_index, friends_index, d):
        r = self.R(friends_index)
        c = self.C(users_index)
        d = self.D(items_index)
        feedback = (r * c * d).sum(1)
        return feedback
        

    def __call__(self, *args):
        return self.forward(*args)


    def predict_sing(self, users_index, items_index):
        preds = self.forward_sing(users_index, items_index)
        return preds
    
    
    

