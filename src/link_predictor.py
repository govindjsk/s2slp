import argparse
import data_preparer
import networkx as nx
import numpy as np
import os
import pandas as pd
import pdb
import pickle

from collections import defaultdict
from data_preparer import RawData, S2SLPData
from itertools import product
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

def get_lp_scores(v, v_, nbrs, nbrs_):
    nbrs_v = set(nbrs.get(v, set())) # Subset of V'
    nbrs_v.discard(v_)

    nbrs_nbrs_v = set() # Subset of V
    for nv in nbrs_v: # n_v is an element of V'
        nbrs_nv = set(nbrs_[nv]) # Subset of V
        nbrs_nbrs_v.update(nbrs_nv)
    nbrs_v_ = set(nbrs_.get(v_, set())) # Subset of V
    nbrs_v_.discard(v)
    nbrs_nbrs_v_ = set() # Subset of V'
    for nv_ in nbrs_v_: # n_v_ is an element of V
        nbrs_nv_ = set(nbrs[nv_]) # Subset of V'
        nbrs_nbrs_v_.update(nbrs_nv_)
    cn = nbrs_nbrs_v.intersection(nbrs_v_)
    cn_ = nbrs_nbrs_v_.intersection(nbrs_v)
    scores = {'cn': len(cn), 'cn_': len(cn_), 'cn_mean': (len(cn)+len(cn_))/2}
    return scores



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
    B_nbrs, B_nbrs_ = get_bipartite_nbrs(B)
    A_nbrs, A_nbrs_ = get_bipartite_nbrs(A)
    elements = incidence_to_hyperedges(S, _type=dict)
    elements_ = incidence_to_hyperedges(S_, _type=dict)
    
    def calculate_lp_scores(pairs, labels):
        results = {}
        for (v, v_), l in (list(zip(pairs, labels))):
            result = {}
            scores = get_lp_scores(v, v_, B_nbrs, B_nbrs_)
            result.update({'B_{}'.format(a): s for a, s in scores.items()})

            result.update({'A_{}'.format(a): [] for a in scores})
            f_v = set(elements[v])
            f_v_=set(elements_[v_]) # Set of nodes incident to hyperedge ids v and v_
            count = 0
            for i, j in product(f_v, f_v_): # i \in f_v, j \in f_v_
                scores = get_lp_scores(i, j, A_nbrs, A_nbrs_)
                _ = {result['A_{}'.format(a)].append(s) for a, s in scores.items()}
                count += 1
            for a in scores:
                result['min_A_{}'.format(a)] = min(result['A_{}'.format(a)])
                result['max_A_{}'.format(a)] = max(result['A_{}'.format(a)])
                result['avg_A_{}'.format(a)] = np.mean(result['A_{}'.format(a)])
                del result['A_{}'.format(a)]
            result.update({'label': l})
            results.update({(v, v_): result})
        df = pd.DataFrame(results).T
        return df
    train_df = calculate_lp_scores(train_pairs, train_labels)
    test_df = calculate_lp_scores(test_pairs, test_labels)
    
    
    return train_df, test_df

def get_perf_results(data_home, data_name, num_exp, silent = False):
    train_aucs = defaultdict(list)
    test_aucs = defaultdict(list)
    for i in tqdm(range(num_exp)):
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
                                                   round(np.std(aucs), 4)])) for a, aucs in test_aucs.items()}}
    perf_results_df = pd.DataFrame(perf_results)
    return perf_results_df

def init_args():
    parser = argparse.ArgumentParser(description='Baseline Link Predictor for S2SLP')
    parser.add_argument('--data-home', default='/home2/e1-313-15477/govind/s2slp/data/', help='network path')
    parser.add_argument('--data-name', default='mag-acm-full', help='network name')
    parser.add_argument('--num-exp', type=int, default=10, help='number of experiments for statistical significance')
    parser.add_argument('--silent', action='store_true', default=False, help='whether or not to go with GCC')
    args = parser.parse_args()
    return args

def main():
    args = init_args()
    perf_results = get_perf_results(args.data_home, args.data_name, args.num_exp, args.silent)
    print(perf_results)

if __name__ == '__main__':
    main()
    