import numpy as np
from collections import defaultdict
import tqdm
from matplotlib import pyplot as plt
import networkx as nx
import random

MAX_DISPLAY_NODES = 100

def compare_iterables(iter1, iter2):
    equal_flags = []
    for x, y in zip(iter1, iter2):
        try:
            equal_flags.append((x!=y).nnz==0)
        except AttributeError:
            equal_flags.append(x==y)
        except ValueError:
            equal_flags.append(np.array_equal(x, y))
    if all(equal_flags):
        return True
    else:
        print(equal_flags)
        return False
    
def incidence_to_hyperedges(S, silent_mode=True, _type=set):
    I, J = S.nonzero()
    hyperedges = defaultdict(set)
    indices = list(zip(I, J))
    if not silent_mode:
        print('Converting incidence matrix to hyperedge {} for faster processing...'.format(_type))
    for i, j in (tqdm(indices) if not silent_mode else indices):
        hyperedges[j].add(i)
    if _type == set:
        return set(map(frozenset, hyperedges.values()))
    elif _type == list:
        return set(map(frozenset, hyperedges.values()))
    elif _type == dict:
        return {i: set(f) for i, f in hyperedges.items()}
    return hyperedges

def get_bipartite_nbrs(B):
    nbrs = incidence_to_hyperedges(B.T, _type=dict)
    nbrs_ = incidence_to_hyperedges(B, _type=dict)
    return nbrs, nbrs_

def plot_bipartite_nw(B):
    n, n_ = B.shape
    if max([n, n_]) > MAX_DISPLAY_NODES:
        print('WARNING: Bipartite graph too large to be drawn. Drawing a subgraph instead.')
        if n > MAX_DISPLAY_NODES:
            V = random.sample(range(n), MAX_DISPLAY_NODES)
            B = B[V, :]
            n = MAX_DISPLAY_NODES
        if n_ > MAX_DISPLAY_NODES:
            V_= random.sample(range(n_), MAX_DISPLAY_NODES)
            B = B[:, V_]
            n_ = MAX_DISPLAY_NODES
    my_dpi = 72
    width = 0.5+np.log(max([n, n_]))
    height = min([(2**16 - 1)/my_dpi, 0.5+0.4*max([n, n_])])
    fig, ax = plt.subplots(figsize = (width, height))
    G = nx.bipartite.from_biadjacency_matrix(B)
    pos = nx.drawing.layout.bipartite_layout(G, range(n))
    node_labels=dict([(i, str(i)) for i in range(n)] + [(i + n, str(i)) for i in range(n_)])
#     print(node_labels)
    nx.draw(G, pos, with_labels=True, node_color=['yellow']*n + ['cyan']*n_, ax=ax, labels = node_labels)