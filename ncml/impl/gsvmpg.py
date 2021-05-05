from .base import BaseClassifier, BaseRegressor
from .loss import Logistic, SquaredLoss
from .penalty import SCAD, MCP, LASSO
import numpy as np
from numpy.linalg import norm
import random
import time


class _BaseGSVMPG(object):
    """
    base class for algorithm generalized stochastic variable metric
    proximal gradient

    """
    def _get_loss(self, X, y):
        raise NotImplementedError()

    def _get_penalty(self):
        raise NotImplementedError()

    def _fit(self, X, y):
        loss = self._get_loss(X, y)
        penalty = self._get_penalty()
        n_samples, n_features = X.shape

        # initialization
        if self.warm_start:
            if self.coef_ is None:
                w_init = np.zeros(n_features)
            else:
                w_init = self.coef_
        w_k = w_init.copy()
        w_t = w_init.copy()
        coordinate_stepsize = 1e-3

        for k in range(self.outer_iter):
            for t in range(self.inner_iter):
            # coordinate descent
            while True:
                idx = random.choice(list(range(n_features)))


