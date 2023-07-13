import jellyfish as jf
import Levenshtein as lev
import numpy as np

def soft_inter(A, B, sim_fun, theta):
    return set([item for sublist in [(s, t) for s in A for t in B if sim_fun(s, t) >= theta] for item in sublist])


def soft_injection_index(A, B, sim_fun, theta):
    soft_inter_A_B = soft_inter(A, B, sim_fun, theta)
    if A:
      return len(soft_inter_A_B.intersection(A)) / len(A)
    else:
      return 0


def similar_relatedness(A, B, N, sim_fun, theta=1):
    la = len(A)
    lb = len(B)
    lmin = min(la, lb)
    if lmin == 0:
        return 0
    else:
        if theta != 1:
            inter = soft_inter(A, B, sim_fun, theta)
        else:
            inter = A.intersection(B)
        if inter:
            lmax = max(la, lb)
            linter = len(inter)
            return 1 - (np.log(lmax / linter)) / (np.log(N / lmin))
        else:
            return 0


def coh_score(IN_e1, IN_e2, N=1e5):
    return similar_relatedness(IN_e1, IN_e2, N, None, 1)