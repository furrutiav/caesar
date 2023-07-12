from class_wikiwrapper import WikiWrapper
from class_cohm import COHM
from token_metrics import similar_relatedness, soft_inter
from string_metrics import sim_trans_str_measure
import pickle


def get_max(sim_fun, k, r_red, key=""):
    m = ["", 0]
    for c in r_red:
        t = c
        if key:
            t = c[key]
        nt = t.lower().strip()
        sl = sim_fun(k, nt)
        if sl > m[1]:
            m[1] = sl
            m[0] = t
    return m


class LinkScore(object):
    def __init__(self, cohm: COHM):
        self.params = None
        self.norm = 0
        self.cohm = cohm
        self.ww = cohm.ww
        self.single_str_score = {}
        self.single_ctx_score = {}
        self.sim_str_measure = sim_trans_str_measure

    def add_params(self, params: dict):
        self.params = params
        self.norm = sum(params.values())

    def score(self, entities_context, entities_mentioned, X):
        a1 = self.params["a1"]
        a3 = self.params["a3"]
        a2 = self.params["a2"]

        sum_str = sum([self.get_single_str_score(entities_mentioned[i], n)["max"] for i, n in enumerate(X)])
        sum_ctx = sum([self.get_single_ctx_score(entities_context, n)["union"] for n in X])

        sum_coh = 0
        N_coh = 0
        for i, ni in enumerate(X):
            for nj in X[i:]:
                sum_coh += self.cohm.get_coh(ni, nj)
                N_coh += 1

        print((1 / self.norm) * a1 * (1 / len(X)) * sum_str, (1 / self.norm) * a2 * (1 / len(X)) * sum_ctx,
              (1 / self.norm) * a3 * (1 / N_coh) * sum_coh)
        return (1 / self.norm) * (
                a1 * (1 / len(X)) * sum_str + a2 * (1 / len(X)) * sum_ctx + a3 * (1 / N_coh) * sum_coh)

    def get_single_str_score(self, entity_mentioned, entity_named):
        m, n = entity_mentioned, entity_named
        if m in self.single_str_score.keys():
            if n in self.single_str_score[m].keys():
                return self.single_str_score[m][n]
        n_context = self.ww.get_context(n)
        sim_str_measure = self.sim_str_measure
        # title
        sim_title = sim_str_measure(m.lower().strip(), n.lower())
        # redirect
        _, sim_redirect = get_max(sim_str_measure, m.lower().strip(), n_context["redirects"], key="title")
        # bold
        entity_bold, sim_bold = get_max(sim_str_measure, m.lower().strip(), n_context["bold"]["ix"], key="")
        rank_bold = 0
        if entity_bold:
            first_time = n_context["bold"]["ix"][entity_bold][0]
            rank_bold = n_context["bold"]["dic"][first_time]["rank"]
        # general
        sim = {"title": sim_title, "redirect": sim_redirect, "bold": sim_bold * rank_bold}
        sim["max"] = max(sim.values())
        if m not in self.single_str_score.keys():
            self.single_str_score[m] = {n: sim}
        elif n not in self.single_str_score[m].keys():
            self.single_str_score[m][n] = sim
        return sim

    def get_single_ctx_score(self, entities_context, entity_named):
        m_context, n = set(entities_context), entity_named
        m = str(entities_context)

        if m in self.single_ctx_score.keys():
            if n in self.single_ctx_score[m].keys():
                return self.single_ctx_score[m][n]
        n_context = self.ww.get_context(n)
        sim_str_measure = self.sim_str_measure
        # hlight
        hlight_context = set(n_context["hlight"]["ix"].keys())
        sim_hlight = similar_relatedness(m_context, hlight_context, 60, sim_str_measure, 0.7)
        # categories
        categories_context = set([c["title"].replace("Category:", "").lower().strip() for c in n_context["categories"]])
        sim_categories = similar_relatedness(m_context, categories_context, 60, sim_str_measure, 0.7)
        # ilinks
        ilinks_context = set([c["title"].lower().strip() for c in n_context["ilinks"]])
        sim_ilinks = similar_relatedness(m_context, ilinks_context, 60, sim_str_measure, 0.7)
        # olinks
        olinks_context = set([c["title"].lower().strip() for c in n_context["olinks"]])
        sim_olinks = similar_relatedness(m_context, olinks_context, 60, sim_str_measure, 0.7)
        # general
        sim = {"hlight": sim_hlight,
               "categories": sim_categories,
               "ilinks": sim_ilinks,
               "olinks": sim_olinks,
               "union": len(
                   soft_inter(
                       m_context,
                       olinks_context.union(ilinks_context).union(categories_context).union(hlight_context),
                       sim_str_measure,
                       0.7).intersection(m_context)) / 60}
        if m not in self.single_ctx_score.keys():
            self.single_ctx_score[m] = {n: sim}
        elif n not in self.single_ctx_score[m].keys():
            self.single_ctx_score[m][n] = sim
        return sim

    def diff_score(self, entities_context, entities_mentioned, X, Y, k):
        a1 = self.params["a1"]
        a3 = self.params["a3"]
        a2 = self.params["a2"]

        mk = entities_mentioned[k]

        str_X = self.get_single_str_score(mk.lower(), X[k])["max"]
        str_Y = self.get_single_str_score(mk.lower(), Y[k])["max"]

        ctx_X = self.get_single_ctx_score(entities_context, X[k])["union"]
        ctx_Y = self.get_single_ctx_score(entities_context, Y[k])["union"]

        sum_X = sum([self.cohm.get_coh(X[k], Xi) for i, Xi in enumerate(X) if i != k])
        sum_Y = sum([self.cohm.get_coh(Y[k], Yi) for i, Yi in enumerate(Y) if i != k])

        return (1 / self.norm) * (
                    a1 * (str_X - str_Y) + a2 * (ctx_X - ctx_Y) + a3 * (1 / len(entities_mentioned)) * (
                        sum_X - sum_Y))

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def load(self, file_name):
        with open(file_name, 'rb') as file:
            new_self = pickle.load(file)
        for k, v in vars(new_self).items():
            self.__dict__[k] = v