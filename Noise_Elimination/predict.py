#/usr/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
from random import sample
from random import seed
from random import shuffle
import copy
from tqdm import tqdm
from cost import count3gametes
from time import time
from keras import backend as K
from data import data
seed(30)


import numpy as np

class Mat:
    def __init__(self, m):
        self.mat = m
        self.nCells = m.shape[0]
        self.nMuts = m.shape[1]
        self.a01, self.a10, self.a11, self.aVio = tuple([np.zeros(self.nMuts*(self.nMuts-1)//2, dtype = np.int16) for i in range(4)])
        i = 0
        for c in range(self.nMuts):
            for c1 in range(c):
                for r in range(self.nCells):
                    if m[r][c]:
                        if m[r][c1]:
                            self.a11[i] += 1
                        else:
                            self.a10[i] += 1
                    elif m[r][c1]:
                        self.a01[i] += 1
                i += 1
        self.aVio = self.a01 * self.a10 * self.a11
        self.nVio = np.sum(self.aVio)

    def update(self, pos):
        r = pos // self.nMuts
        c = pos % self.nMuts
        i = c*(c-1)//2
        a01, a10 = self.a01, self.a10

        for c1 in range(self.nMuts):
            if c1 == c:
                a01, a10 = a10, a01
            else:
                old_vio = self.aVio[i]
                if self.mat[r][c]:
                    if self.mat[r][c1]:
                        self.a11[i] -= 1
                        a01[i] += 1
                    else:
                        a10[i] -= 1
                elif self.mat[r][c1]:
                    a01[i] -= 1
                    self.a11[i] += 1
                else:
                    a10[i] += 1

                self.aVio[i] = a01[i] * a10[i] * self.a11[i]
                self.nVio += self.aVio[i] - old_vio

            i += (1 if c1 < c else c1)
        self.mat[r][c] = 1 - self.mat[r][c]

"""
m = np.array([[0,1,1],[1,0,0],[1,1,0]])
cm = Mat(m)
print(cm.nVio)
cm.update(0,1)
print(cm.nVio)
print(cm.aVio)
cm2 = Mat(cm.mat)
print(cm2.nVio)
print(cm2.aVio)
print(cm.mat)
print(cm.a01)
print(cm2.a01)
print(cm.a10)
print(cm2.a10)
print(cm.a11)
print(cm2.a11)
"""

class MatFlipProb:
    def __init__(self, m):
        self.mat = m
        self.flips = np.zeros(m.nCells * m.nMuts, dtype=bool)
        self.new_flip = -1
        self.log_prob = 0.0
        self.nll = 0.0
        self.f_0_to_1 = 0
        self.f_1_to_0 = 0
    def update(self):
        self.flips = copy.copy(self.flips)
        self.flips[self.new_flip] = True
        self.mat = copy.deepcopy(self.mat)
        self.mat.update(self.new_flip)
        return self.mat.nVio


def solve(model_actor, config, n_hidden, matrices):

    matrices_p_t = np.asarray(matrices[0])
    matrices_n_t = np.asarray(matrices[1])
    assert np.shape(matrices_n_t) == np.shape(matrices_p_t)
    nMats = np.shape(matrices_n_t)[0]

    N00_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
    N11_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
    N10_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
    N01_NLL_o = np.zeros((nMats, 1), dtype = np.float64)
    NLL_o = np.zeros((nMats, 1), dtype = np.float64)
    NLL_init = np.zeros(nMats, dtype = np.float64)

    fp_fn = np.zeros((nMats, config.nCells, config.nMuts), dtype = np.float32)

    output_ = np.zeros((nMats, 15), dtype = np.float64)

    for k in range(np.shape(matrices_p_t)[0]):
        fp_fn[k, matrices_n_t[k,:,:] == 1] = config.alpha
        fp_fn[k, matrices_n_t[k,:,:] == 0] = config.beta
        
        N01_o_ = np.sum(matrices_n_t[k,:,:] - matrices_p_t[k,:,:] == -1) 
        N10_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == -1)
        N11_o_ = np.sum(matrices_p_t[k,:,:] + matrices_n_t[k,:,:] == 2)
        N00_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == 0) - N11_o_
        
        output_[k,4] = N01_o_ # f_0_to_1_o
        output_[k,6] = N10_o_ # f_1_to_0_o
        output_[k,9] = N00_o_
        output_[k,10] = N11_o_

        N00_NLL_o[k, 0] = N00_o_*np.log(1/(1-config.beta))
        N11_NLL_o[k, 0] = N11_o_*np.log(1/(1-config.alpha))
        N01_NLL_o[k, 0] = N01_o_*np.log(1/config.beta)
        N10_NLL_o[k, 0] = N10_o_*np.log(1/config.alpha)
        output_[k,2] = np.sum([N00_NLL_o[k, 0], N11_NLL_o[k, 0], N01_NLL_o[k, 0], N10_NLL_o[k, 0]])
        NLL_init[k] = (N00_o_ + N01_o_) * np.log(1/(1-config.beta)) + (N11_o_ + N10_o_) * np.log(1/(1-config.alpha))
        output_[k,8] = N01_o_ + N10_o_
             
    l = []
    for i in range(config.nCells):
        for j in range(config.nMuts):
            l.append([i,j])
    l = np.asarray(l)

    max_length = config.nCells * config.nMuts
    a = np.expand_dims(matrices_n_t.reshape(-1, max_length),2)
    b = np.expand_dims(fp_fn.reshape(-1, max_length),2)
    x = np.tile(l,(nMats,1,1))
    c = np.squeeze(np.concatenate([x,b,a], axis = 2))
    #d = np.asarray([np.take(c[i,:,:],np.random.permutation(c[i,:,:].shape[0]),axis=0,out=c[i,:,:]) for i in range(np.shape(c)[0])])
    
    f_input = np.random.randn(config.batch_size, n_hidden)

    NLL_change = (np.log(1/config.beta-1), np.log(1/config.alpha-1))

    for j in tqdm(range(nMats)): # num of examples
        start_t = time()

        m = Mat(matrices_n_t[j])
        q = [MatFlipProb(m)]
        output_[j,12] = m.nVio # original number of violations, V_o
        if m.nVio == 0: continue
        input_batch = np.tile(c[j,:,:],(config.batch_size,1,1))

        best_nll = float("inf")
        best_sol = None
        # beam search, beam size = batch size
        for depth in range(max_length//10):
            print(len(q), ':')
            logps = model_actor.predict({'main_input': input_batch, 'f_input':f_input}, batch_size = config.batch_size)

            temp_q = []
            i = 0
            for mfp_old in q:
                #print(mfp_old.mat.nVio)
                for l in range(max_length):
                    mfp = copy.copy(mfp_old)
                    mfp.new_flip = l
                    mfp.log_prob += 1 #logps[i][l] #np.log(1/(max_length-depth))#logps[i][l]
                    entry = mfp.mat.mat[l//mfp.mat.nMuts][l%mfp.mat.nMuts]
                    mfp.nll += NLL_change[entry]
                    if entry:
                        mfp.f_1_to_0 += 1
                    else:
                        mfp.f_0_to_1 += 1
                    temp_q.append(mfp)
                i += 1
            temp_q.sort(key = lambda x: x.log_prob, reverse = True)            
            shuffle(temp_q)

            q = []
            i = 0
            for mfp in temp_q:
                if i >= config.batch_size:
                    break
                elif not mfp.flips[mfp.new_flip]:
                    #print(mfp.mat.nVio, mfp.log_prob)
                    if mfp.update():
                        q.append(mfp)
            #            print(mfp.mat.nVio, ' ', len(q))
                        m = input_batch[i,:,3] = np.reshape(mfp.mat.mat,-1)
                        input_batch[i,:,2] = np.where(m, config.alpha, config.beta)
                        i += 1
                    elif mfp.nll < best_nll: # no violation, solution found
                        best_nll = mfp.nll
                        best_sol = mfp
            # print(len(q))

        dur_t = time() - start_t
        print(j,'!\n')

        output_[j,0] = dur_t
        output_[j,1] = NLL_init[j] + best_nll
        if best_sol is not None:
            output_[j,3] = best_sol.f_0_to_1
            output_[j,5] = best_sol.f_1_to_0
        output_[j,7] = output_[j,3] + output_[j,5]

    output_[:,11] = 0 # no violations after flipping according to search result
    output_[:,13] = config.alpha # fp
    output_[:,14] = config.beta # fn
    
    df = pd.DataFrame(output_, index = ["test" + str(k) for k in range(nMats)], \
                     columns = ["time", "NLL_rl", "NLL_o", "f_0_to_1_rl", "f_0_to_1_o", "f_1_to_0_rl", "f_1_to_0_o", \
                                "n_f_rl", "n_f_o", "N00_o", "N11_o", "V_rl", "V_o", "fp", "fn"])
    df.to_csv(config.output_dir + '/test_{nCells}x{nMuts}.csv'.format(nCells = config.nCells, nMuts = config.nMuts), sep = ',')

