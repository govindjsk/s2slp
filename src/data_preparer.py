import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.sparse as ssp

from collections import namedtuple
from scipy.sparse import csr_matrix
from tqdm import tqdm

silent_flag = True
RawData = namedtuple('RawData', ['S', 'S_', 'B'])
S2SLPData = namedtuple('S2SLPData', ['S', 'S_', 'B',
                                     'train_pos', 'train_neg',
                                     'test_pos', 'test_neg',
                                     'node_emb', 'edge_emb'])

def load_bipartite_hypergraph(data_params):
    id_p_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_label_file']), sep='\t', header=None)
    id_a_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['u_label_file']), sep='\t', header=None)
    id_a_map = dict(zip(id_a_map[0], id_a_map[1]))
    id_k_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['v_label_file']), sep='\t', header=None)
    id_k_map = dict(zip(id_k_map[0], id_k_map[1]))
    p_a_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_u_list_file']), sep=':', header=None)
    p_k_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_v_list_file']), sep=':', header=None)
    n_p, na, nk = len(id_p_map), len(id_a_map), len(id_k_map)
    pos_A = list(map(lambda x: list(map(int, x.split(','))), p_a_list_map[1]))
    pos_B = list(map(lambda x: list(map(int, x.split(','))), p_k_list_map[1]))    
    # I, J, V: row, col, value of author-hypergraph
    # I_, J_, V_: row, col, value of keyword-hypergraph
    # I_B, J_B, V_B: row, col, value of author_hyperedge-keyword_hyperedge link
    I, J, V, I_, J_, V_, I_B, J_B, V_B = [], [], [], [], [], [], [], [], []
    U_set, V_set = set(), set()
    u_map, v_map = {}, {}
    j_u, j_v = -1, -1
    for u,v in zip(pos_A,pos_B):
        u, v = frozenset(u), frozenset(v)
        if u not in U_set:
            j_u+=1
            U_set.add(u)
            u_map[u]=j_u
            I.extend(list(u))
            J.extend([j_u]*len(u))
            V.extend([1]*len(u))
        if v not in V_set:
            j_v+=1
            V_set.add(v)
            v_map[v]=j_v
            I_.extend(list(v))
            J_.extend([j_v]*len(v))
            V_.extend([1]*len(v))
        I_B.append(u_map[u])
        J_B.append(v_map[v])
        V_B.append(1)
    n, m, n_, m_ = max(I) + 1, len(U_set), max(I_) + 1, len(V_set)
    S = csr_matrix((V, (I, J)), shape=(n, m))
    S_ = csr_matrix((V_, (I_, J_)), shape=(n_, m_))
    B = csr_matrix((V_B, (I_B, J_B)), shape=(m, m_))    
    return S,S_,B

def sample_neg_bip(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    '''
    Note that net is NOT a symmetric matrix!
    '''
    # sample positive links for train/test
    row, col, _ = ssp.find(net)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    m,n = net.shape
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, m-1), random.randint(0, n-1)
        if net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (np.array(neg[0][:train_num]), np.array(neg[1][:train_num]))
    test_neg = (np.array(neg[0][train_num:]), np.array(neg[1][train_num:]))
    return train_pos, train_neg, test_pos, test_neg

def read_raw_data(raw_data_params, silent = False):
    S, S_, B = load_bipartite_hypergraph(raw_data_params)
    data = RawData(S, S_, B)
    if not silent:
        print("Data:\n\tS : {}x{} (nnz={})\n"
              "\tS': {}x{} (nnz={})\n"
              "\tB : {}x{} (nnz={})\n".format(*data.S.shape, data.S.nnz,
                                                        *data.S_.shape, data.S_.nnz,
                                                        *data.B.shape, data.B.nnz))
    if raw_data_params['gcc']:
        data = filter_on_gcc(data, silent = silent)
    return data

def prepare_s2slp_data(raw_data, s2slp_data_params, silent = False):
    train_pos, train_neg, test_pos, test_neg = sample_neg_bip(raw_data.B, s2slp_data_params['test_ratio'],
                                                              s2slp_data_params['max_train_num'])
    A = raw_data.B.copy()
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A.eliminate_zeros()
    s2slp_data = S2SLPData(raw_data.S, raw_data.S_, A,
                           train_pos, train_neg, test_pos, test_neg,
                           None, None)
    if not silent:
        print("S2SLPData:\n\tS : {}x{} (nnz={})\n"
              "\tS': {}x{} (nnz={})\n"
              "\tB : {}x{} (nnz={})\n"
              "\tTr: {}+ {}-\tTe: {}+ {}-".format(*s2slp_data.S.shape, s2slp_data.S.nnz,
                                                        *s2slp_data.S_.shape, s2slp_data.S_.nnz,
                                                        *s2slp_data.B.shape, s2slp_data.B.nnz,
                                                        len(s2slp_data.train_pos[0]), len(s2slp_data.train_neg[0]),
                                                        len(s2slp_data.test_pos[0]), len(s2slp_data.test_neg[0])))
    return s2slp_data


def filter_on_gcc(raw_data, silent = False):
    S, S_, B = raw_data
    G = nx.bipartite.from_biadjacency_matrix(B)
    ccs = nx.connected_components(G)
    gcc = max(ccs, key = lambda x: len(x))
    gcc, gcc_ = list(sorted({x for x in gcc if x < B.shape[0]})), \
                list(sorted({x-B.shape[0] for x in gcc if x >= B.shape[0]}))
    set_gcc, set_gcc_ = set(gcc), set(gcc_)
    B = B[gcc, :][:, gcc_]
    S = S[:, gcc]
    S = S[(S.sum(axis=1) != 0).nonzero()[0], :]
    S_ = S_[:, gcc_]
    S_ = S_[(S_.sum(axis=1) != 0).nonzero()[0], :]
    raw_data_gcc = RawData(S, S_, B)
    if not silent:
        print("Filtering on GCC...")
        print("Data:\n\tS : {}x{} (nnz={})\n"
              "\tS': {}x{} (nnz={})\n"
              "\tB : {}x{} (nnz={})\n".format(*raw_data_gcc.S.shape, raw_data_gcc.S.nnz,
                                                        *raw_data_gcc.S_.shape, raw_data_gcc.S_.nnz,
                                                        *raw_data_gcc.B.shape, raw_data_gcc.B.nnz))
    return raw_data_gcc

def main():
    data_home = '/home2/e1-313-15477/govind/s2slp/data/'
    data_name = 'main_data'
#     data_name = 'sample_data'
    data_path = os.path.join(data_home, data_name)
    num_experiments = 10
    raw_data_params = {'home_path': data_path,
                       'r_label_file': 'id_p_map.txt',
                       'u_label_file': 'id_a_map.txt',
                       'v_label_file': 'id_k_map.txt',
                       'r_u_list_file': 'p_a_list_train.txt',
                       'r_v_list_file': 'p_k_list_train.txt',
                       'emb_pkl_file': 'nodevectors.pkl',
                       'gcc': True}
    s2slp_data_params = {'test_ratio': 0.3,
                         'max_train_num': None
                         }

    raw_data = read_raw_data(raw_data_params, silent = silent_flag)
    pickle.dump(raw_data, open(os.path.join(data_path, '{}.raw'.format(data_name)), 'wb'))
    for i in tqdm(range(num_experiments)):
        s2slp_data = prepare_s2slp_data(raw_data, s2slp_data_params, silent = silent_flag)
        pickle.dump(s2slp_data, open(os.path.join(data_path, '{}.{}.s2slp'.format(data_name, i)), 'wb'))


if __name__ == '__main__':
    main()
