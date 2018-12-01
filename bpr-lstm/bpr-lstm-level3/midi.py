# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:11:13 2018

@author: DELL
"""

import json
import numpy as np



class Midi(object):
    def __init__(self, midi_path, tick = 300, length = 100):
        self.length = length
        self.midi_path = midi_path
        self.tick = tick
        self.midi_seq, self.max_pitch, self.min_len, self.max_len = self.preprocess_midi(self.tick)

    def load_midi(self):
        """
        load midi file
        """
        print("load file: ", self.midi_path)
        sing_dict = {}
        i = 1
        with open(self.midi_path, 'r') as f:
            for line in f.readlines():
                sid, midi = line.split(" ")
                #print(i, sid)
                midi = json.loads(midi)
                sing_dict[sid] = midi
                i += 1
        return sing_dict

    
    def preprocess_midi(self, tick):
        """
        minus pitch with 12*k
        convert to a sequence of pitch per tick(ms)
        """
        midi_seq = {}
        max_pitch = 0
        min_len = 10000
        max_len = 0
        midi_dict = self.load_midi()
        for sid, midi in midi_dict.items():
            cents_min = midi['cents_min']
            k = int(cents_min/12)
            minus = 12 * k
            pitch = []
            time = midi['data'][0][0]
            i = 0
            for t, delta, p in midi['data']:
                if time + i*tick < t:
                    # max_pitch is 78
                    # so let null be 79
                    # pitch encoding dim should be 80: 0-79
                    pitch.append(79)
                    i += 1
                while (time + i*tick) < (t+delta):
                    p_new = p - minus
                    pitch.append(p_new)
                    if p_new > max_pitch:
                        max_pitch = p_new
                    i += 1

            if len(pitch) > max_len:
                max_len = len(pitch)
            if len(pitch) < min_len:
                min_len = len(pitch)
            while len(pitch) < self.length:
                pitch += pitch
            midi_seq[int(sid)] = pitch[:self.length]
        return midi_seq, max_pitch, min_len, max_len
    
    def get_batch(self, sid_list):
        midi_batch = [self.midi_seq[sid] for sid in sid_list]
        return midi_batch


            
    


    
