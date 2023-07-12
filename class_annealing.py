from class_linkscore import LinkScore
import numpy as np


class Annealing(object):
    def __init__(self, reference, base, space, ls: LinkScore, nf, beta):
        self.reference = reference
        self.base = base
        self.space = space
        self.ls = ls

        self.m = len(space)
        self.shape = [len(r) for r in space]
        print(self.shape)

        self.nf = nf
        self.beta = beta

        self.CM = []
        self.X = []

    def estado_inicial(self):
        return np.array([np.random.randint(0, self.shape[i]) for i in range(self.m)])

    def Trans(self, x, u, next=False):
        if next:
            return np.array(x)
        i = int(np.floor(self.m * u))
        if self.shape[i] == 1:
            return self.Trans(x, 1 - u, next=True)
        yi = np.random.randint(0, self.shape[i])
        y = x.copy()
        if yi == x[i]:
            yi = (yi + 1) % self.shape[i]
        y[i] = yi
        return np.array(y)

    def coef_R(self, x, y, beta):
        k = np.argwhere(True == (x != y))[0][0]
        X = [self.space[i][xi] for i, xi in enumerate(x)]
        Y = [self.space[i][yi] for i, yi in enumerate(y)]
        dif = self.ls.diff_score(self.reference, self.base, Y, X, k)
        return np.exp(-beta * (-dif))

    def MCMC(self, u, v, save_rate=100):
        x0 = self.estado_inicial()
        CM = [x0]
        xn_1 = x0
        for n, un in enumerate(u):
            vn = v[n]
            y = self.Trans(xn_1, vn)
            if un <= self.coef_R(xn_1, y, self.beta(n)):
                xn_1 = y
            if n % save_rate == 0 or n == self.nf - 1:
                CM.append(xn_1)
        self.CM = CM
        self.X = CM[-1]

    def H(self, x):
        X = [self.space[i][xi] for i, xi in enumerate(x)]
        return -self.ls.score(self.reference, self.base, X)