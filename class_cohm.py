from class_wikiwrapper import WikiWrapper
import numpy as np
from token_metrics import coh_score
import pickle

"""
Class COHM (Coherence Manager)
"""


def links_to_set(L):
    return set(np.unique(np.array(L).astype(str)))


class COHM():
    def __init__(self, ww: WikiWrapper):
        self.coh = {}
        self.IN = {}
        self.ww = ww

    def get_coh(self, e1, e2, N=1e6):
        s1, s2 = sorted([e1, e2])
        if s1 in self.coh.keys():
            if s2 in self.coh[s1].keys():
                return self.coh[s1][s2]
        dic_IN = {s1: set(), s2: set()}
        for s in [s1, s2]:
            if s in self.IN.keys():
                dic_IN[s] = self.IN[s]
            else:
                IN_s = links_to_set(self.ww.get_iolinks(s)["links"])
                dic_IN[s] = IN_s
                self.IN[s] = IN_s
        coh_s1_s2 = coh_score(dic_IN[s1], dic_IN[s2], N)
        if s1 in self.coh.keys():
            self.coh[s1][s2] = coh_s1_s2
        else:
            self.coh[s1] = {s2: coh_s1_s2}
        return coh_s1_s2

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def load(self, file_name):
        with open(file_name, 'rb') as file:
            new_self = pickle.load(file)
        for k, v in vars(new_self).items():
            self.__dict__[k] = v