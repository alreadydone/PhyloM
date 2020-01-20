import numpy as np
import pandas as pd
from random import uniform
from copy import deepcopy
import itertools
import os
import shutil


def count3gametes(matrix,nCells, nMuts):
    columnPairs = list(itertools.permutations(range(nMuts), 2))
    nColumnPairs = len(columnPairs)
    columnReplicationList = np.array(columnPairs).reshape(-1)
    replicatedColumns = matrix[:, columnReplicationList].transpose()
    x = replicatedColumns.reshape((nColumnPairs, 2, nCells), order="A")
    col10 = np.count_nonzero( x[:,0,:]<x[:,1,:]     , axis = 1)
    col01 = np.count_nonzero( x[:,0,:]>x[:,1,:]     , axis = 1)
    col11 = np.count_nonzero( (x[:,0,:]+x[:,1,:]==2), axis = 1)
    eachColPair = col10 * col01 * col11
    return np.sum(eachColPair)


def ms(nMats, nCells, nMuts, ms_dir):
    m = []
    matrices = np.zeros((nMats, nCells, nMuts), dtype = np.int8)
    os.mkdir('tmp')
    for i in range(1, nMats + 1):
        cmd ='{ms_dir}/ms {nCells} 1 -s {nMuts} | tail -n {nCells} > {tmp_dir}/m{i}.txt'.format(ms_dir = ms_dir, nCells = nCells, nMuts = nMuts, tmp_dir = 'tmp', i = i)
        os.system(cmd)
    for j in range(1, nMats + 1):
        f = open('tmp/m{j}.txt'.format(j = j), 'r')
        l = [line for line in f]
        l1 = [s.strip('\n') for s in l]
        l2 = np.array([[int(s) for s in q] for q in l1])  # Original matrix
        matrices[j-1,:,:] = l2
        m.append(tuple(l2.flatten()))
        f.close()
    shutil.rmtree('tmp')
    # m1 = list(set(m))
    matrices_u = np.zeros((len(m), nCells, nMuts), dtype = np.int8)
    for k in range(len(m)):
        matrices_u[k,:,:] = np.asarray(m[k]).reshape((nCells, nMuts))
    return matrices_u  # returns all of generated unique matrices
        

def data(nMats, nCells, nMuts, ms_dir, alpha, betta):
    matrices_u = ms(nMats, nCells, nMuts, ms_dir)
    # print(matrices_u)
    matrices_n = []
    matrices_p = []
    for i in range(np.shape(matrices_u)[0]):
        v = 0
        matrix_n = deepcopy(matrices_u[i,:,:].reshape(1, -1))
        while ((count3gametes(matrix_n.reshape(nCells, nMuts), nCells, nMuts) == 0) and (v < nCells*nMuts)):
            matrix_n = deepcopy(matrices_u[i,:,:].reshape(1, -1))
            Zs = np.where(matrix_n  == 0)[1]
            s_fp = np.random.choice([True, False], (1, len(Zs)), p = [alpha, 1 - alpha])  # must be flipped from 0 to 1
            Os = np.where(matrix_n  == 1)[1] 
            s_fn = np.random.choice([True, False], (1, len(Os)), p = [betta, 1 - betta]) # must be flipped from 1 to 0
            matrix_n[0, Zs[np.squeeze(s_fp)]] = 1
            matrix_n[0, Os[np.squeeze(s_fn)]] = 0
            v += 1
            
        if count3gametes(matrix_n.reshape(nCells, nMuts), nCells, nMuts) != 0:
            matrices_n.append(matrix_n.reshape(nCells, nMuts))
            matrices_p.append(matrices_u[i,:,:])
        # print(matrices_n)
        # print(matrices_p)
    return matrices_p, matrices_n
    
    
    
