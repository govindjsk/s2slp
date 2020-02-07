import data_preparer
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle

from collections import defaultdict
from data_preparer import RawData
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook
from utils import plot_bipartite_nw, incidence_to_hyperedges, get_bipartite_nbrs

def get_auc_scores(df):
    algos = list(df.columns)
    algos.remove('label')
    auc_scores = {}
    for a in algos:
        auc_scores[a] = roc_auc_score(df['label'], df[a])
    return auc_scores

def predict_links(prepared_data_params):
    data_home, data_name, i = [prepared_data_params[x] for x in ['data_home', 'data_name', 'i']]
    s2slp_data = pickle.load(open(os.path.join(data_home, data_name,
                                               '{}.{}.s2slp'.format(data_name, i)), 'rb'))
    S, S_, B = s2slp_data[:3]
    train_pairs = list(zip(*s2slp_data.train_pos)) + list(zip(*s2slp_data.train_neg))
    train_labels = [1]*len(s2slp_data.train_pos[0]) + [0]*len(s2slp_data.train_neg[0])
    test_pairs = list(zip(*s2slp_data.test_pos)) + list(zip(*s2slp_data.test_neg))
    test_labels = [1]*len(s2slp_data.test_pos[0]) + [0]*len(s2slp_data.test_neg[0])
    A = S*B*S_.T
    nbrs, nbrs_ = get_bipartite_nbrs(A)
    def get_lp_scores(v, v_):
        nbrs_v = nbrs.get(v, set()) # Subset of V'

        nbrs_nbrs_v = set() # Subset of V
        for nv in nbrs_v: # n_v is an element of V'
            nbrs_nv = nbrs_[nv] # Subset of V
            nbrs_nbrs_v.update(nbrs_nv)
        nbrs_v_ = nbrs_.get(v_, set()) # Subset of V

        nbrs_nbrs_v_ = set() # Subset of V'
        for nv_ in nbrs_v_: # n_v_ is an element of V
            nbrs_nv_ = nbrs[nv_] # Subset of V'
            nbrs_nbrs_v_.update(nbrs_nv_)
        cn = nbrs_nbrs_v.intersection(nbrs_v_)
        cn_ = nbrs_nbrs_v_.intersection(nbrs_v)
        scores = {'cn': len(cn), 'cn_': len(cn_), 'cn_mean': (len(cn)+len(cn_))/2}
        return scores

    train_results = {}
    for (v, v_), l in (list(zip(train_pairs, train_labels))):
        result = {}
        scores = get_lp_scores(v, v_)
        result.update(scores)
        result.update({'label': l})
        train_results.update({(v, v_): result})
    train_df = pd.DataFrame(train_results).T

    test_results = {}
    for (v, v_), l in (list(zip(test_pairs, test_labels))):
        result = {}
        scores = get_lp_scores(v, v_)
        result.update(scores)
        result.update({'label': l})
        test_results.update({(v, v_): result})
    test_df = pd.DataFrame(test_results).T
    return train_df, test_df

def get_perf_results(data_home, data_name):
    train_aucs = defaultdict(list)
    test_aucs = defaultdict(list)
    for i in tqdm(range(10)):
        prepared_data_params = {'data_home': data_home,
                                'data_name': data_name,
                                'i': i}
        train_lp_scores_df, test_lp_scores_df = predict_links(prepared_data_params)
        train_auc_scores = get_auc_scores(train_lp_scores_df)
        test_auc_scores = get_auc_scores(test_lp_scores_df)
        for a in train_auc_scores:
            train_aucs[a].append(train_auc_scores[a])
            test_aucs[a].append(test_auc_scores[a])

    perf_results = {'train': {a: ''.join(map(str, [round(np.mean(aucs), 4), '+-',
                                                   round(np.std(aucs), 4)])) for a, aucs in train_aucs.items()},
                    'test': {a: ''.join(map(str, [round(np.mean(aucs), 4), '+-',
                                                   round(np.std(aucs), 4)])) for a, aucs in test_aucs.items()},}
    perf_results_df = pd.DataFrame(perf_results)
    return perf_results_df

def main():
    data_home = '/home2/e1-313-15477/govind/s2slp/data/'
    data_name = 'main_data'
    perf_results = get_perf_results(data_home, data_name)
    print(perf_results)

if __name__ == '__main__':
    main()
    