
import numpy
import os
import random
import json
import pandas as pd



class Data(object):
    def __init__(self, sing_train_path, sing_test_path, listen_path, batch_size):
        self.sing_train_path = sing_train_path
        self.sing_test_path = sing_test_path
        self.listen_path = listen_path
        self.batch_size = batch_size
        self.song_num = 1000
        self.sing_train, self.sing_test, self.listen, self.usr_index_map,\
        self.usr_index_map_rev, self.common_usr = self.preprocess()
        self.usr_num = len(self.common_usr)
        
        self.usr_listen_stats = numpy.asarray([[u,len(self.listen[u])] for u in range(self.usr_num)])
        self.usr_sing_train_stats = numpy.asarray([[u,len(self.sing_train[u])] for u in range(self.usr_num)])
        self.usr_sing_test_stats = numpy.asarray([[u,len(self.sing_test[u])] for u in range(self.usr_num)])


        
    def load_data(self, data_path):
        """
        load data: (user, song) pairs in the past 30 days
        
        return a dict
        """
        print("load file: ", data_path)
        sing_dict = {}
        with open(data_path, 'r') as f:
            for line in f:
                decodes = json.loads(line)
                if decodes['uid'] not in sing_dict:
                    sing_dict[decodes['uid']] = set()
                sing_dict[decodes['uid']].add(decodes['sid'])
        return sing_dict



    def preprocess(self):
        """
        figure out the common usr of sing train data and listen data as our final usr
        reindex uid by usr_index_map & usr_index_map_rev
        """
        print("preprocess data ...")
        sing_train = self.load_data(self.sing_train_path)
        sing_test = self.load_data(self.sing_test_path)
        listen_data = self.load_data(self.listen_path)
        sing_train_usr = set(sing_train.keys())
        listen_usr = set(listen_data.keys())
        common_usr = sing_train_usr & listen_usr
        usr_num = len(common_usr)
        #print("commen uid:", usr_num)
        usr_index_map = dict(zip(common_usr,range(usr_num)))
        usr_index_map_rev = dict(zip(range(usr_num), common_usr))
        
        reindex_sing_train =dict().fromkeys(range(usr_num), set())
        for uid, sid_set in sing_train.items():
            if uid in common_usr:
                reindex_sing_train[usr_index_map[uid]] = sid_set
        
    
        reindex_sing_test = dict().fromkeys(range(usr_num), set())
        for uid, sid_set in sing_test.items():
            if uid in common_usr:
                new_uid = usr_index_map[uid] 
                reindex_sing_test[new_uid] = sid_set - reindex_sing_train[new_uid] 
        
        reindex_listen = dict().fromkeys(range(usr_num), set())
        for uid, sid_set in listen_data.items():
            if uid in common_usr:
                reindex_listen[usr_index_map[uid]] = sid_set
        return reindex_sing_train, reindex_sing_test, reindex_listen, usr_index_map, usr_index_map_rev, common_usr



    def generate_sing_train_batch(self, sample_ratio = 0.1):
        """
        uniform sampling (uid, observed_sid, unobserved_sid)
        return [batch,[uid, sid_i, sid_j]]
        """
        t = []
        song_list = set(range(self.song_num))
        total_num = 0
        while total_num < self.batch_size:
            u = random.sample(self.sing_train.keys(), 1)[0]
            sample_num = int(sample_ratio * len(self.sing_train[u]) + 1)
            i = random.sample(self.sing_train[u], sample_num)
            j = random.sample(song_list - self.sing_train[u], sample_num)
            t += [[u,i[k],j[k]] for k in range(sample_num)]
            total_num += sample_num
        return numpy.asarray(t)[:self.batch_size, :]



    def generate_listen_train_batch(self, sample_ratio = 0.1):
        """
        uniform sampling (uid, listened_sid, unlistened_sid)
        return [batch,[uid, sid_i, sid_j]]
        """
        t = []
        song_list = set(range(self.song_num))
        total_num = 0
        while total_num < self.batch_size:
            u = random.sample(self.listen.keys(), 1)[0]
            #sample_num = int(sample_ratio * len(self.listen[u]) + 1)
            sample_num = 1
            i = random.sample(self.listen[u], sample_num)
            j = random.sample(song_list - self.listen[u], 1)
            t += [[u,i[k],j[k]] for k in range(1)]
            total_num += sample_num
        return numpy.asarray(t)[:self.batch_size, :]



    def generate_sing_test_batch_total(self, neg_num = 1):
        """
        uniform sampling (uid, listened_sid, unlistened_sid)
        return [batch,[uid, sid_i, sid_j]]
        """
        t = []
        song_list = set(range(self.song_num))
        for uid, sid_set in self.sing_test.items(): 
            sid_num = len(sid_set)
            i = random.sample(self.sing_test[uid], sid_num)
            for m in range(sid_num):
                j = random.sample(song_list - self.sing_test[uid] - self.sing_train[uid], neg_num)
                t += [[uid,i[m],j[k]] for k in range(neg_num)]
        return numpy.asarray(t)[:self.batch_size, :]
    
    def generate_sing_test_batch(self, n, neg_num = 1):
        """
        uniform sampling (uid, listened_sid, unlistened_sid)
        return [batch,[uid, sid_i, sid_j]]
        """
        t = []
        total_num  = 0
        song_list = set(range(self.song_num))
        while(total_num < n):
            u = random.sample(self.sing_test.keys(), 1)[0]
            try:
                i = random.sample(self.sing_test[u], 1)[0]
                j = random.sample(song_list - self.sing_test[u] - self.sing_train[u], 1)[0]
                t.append([u, i, j])
                total_num += 1
            except:
                pass
        return numpy.asarray(t)




