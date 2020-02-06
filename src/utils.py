import numpy as np
from collections import defaultdict
import tqdm

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