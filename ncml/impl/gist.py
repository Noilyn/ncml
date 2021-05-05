from ncml.impl.loss import Logistic, SquaredLoss
from ncml.impl.penalty import SCAD, MCP, LASSO
from ncml.impl.base import BaseClassifier, BaseRegressor
import numpy as np
import time
from numpy.linalg import norm


class _BaseGIST(object):
    """
    Generalize Iterative Shrinkage and Thresholding Algorithm

    Solve the following problem:
        minimize f(x) = l(x) + r(x)
        f(x): bounded below
        l(x): continuously differentiable with Lipschitz continuous gradient
        r(x): non-smooth and non-convex, can be rewritten as the difference
        of two convex function, i.e., r(x) = r_1(x) - r_2(x)
    """
    def _get_loss(self, X, y):
        raise NotImplementedError()

    def _get_penalty(self):
        raise NotImplementedError()

    def _fit(self, X, y, w_init=None):
        loss = self._get_loss(X, y)
        penalty = self._get_penalty()
        n_samples, n_features = X.shape
        l_const = loss.l_const

        # initialization
        if self.warm_start:
            if self.coef_ is None:
                if w_init is None:
                    w_init = np.zeros(n_features)
            else:
                w_init = self.coef_
        else:
            if w_init is None:
                w_init = np.zeros(n_features)

        w_old = w_init.copy()
        w = w_init.copy()
        ss, gm, sig, msize = l_const, 2, 1e-5, 5
        self.obj_pass = [loss.loss(w_old) + penalty.loss(w_old)]
        self.execution_time_pass = [0]
        t_start = time.time()

        for k in range(self.max_iters):
            grad_w = loss.grad(w)
            try:
                if k > 0:
                    d_g = grad_w - grad_w_old
                    d_w = w - w_old
                    ss = d_g.dot(d_w) / d_w.dot(d_w)
                while True:
                    w_prox = penalty.prox(w - grad_w / ss, alpha=ss)
                    psi_y = loss.loss(w_prox) + penalty.loss(w_prox)
                    fmax = max(self.obj_pass[max(0, k-msize+1):(k+1)])
                    # stopping criterion
                    if psi_y <= fmax - sig * ss / 2 * norm(w_prox - w) ** 2:
                        break
                    ss *= gm
                grad_w_old = grad_w
                w_old = w
                w = w_prox

                # profile
                rel = norm(w_old - w) / (1 + norm(w))
                tcost = time.time() - t_start
                self.obj_pass.append(psi_y)
                self.execution_time_pass.append(tcost)
                if self.verbose and (k % 10 == 0 or k == self.max_iters - 1):
                    print(
                        'Iter %d, ss %.3e, obj %.3e, nnz %.3e, rel %.3e, time %.3e'
                        % (k, ss, self.obj_pass[-1], sum(w != 0), rel, tcost))
            except:
                break
            if rel < self.tol or k == (self.max_iters-1):
                self.iter_num = k
                self.val = self.obj_pass[-1]
                self.nnz = sum(w != 0)
                self.execution_time_cost = tcost
                break
        self.coef_ = w


class GISTClassifier(BaseClassifier, _BaseGIST):
    """
    Estimator for learning linear classifiers by GIST
    """
    def __init__(self, loss='logistic', penalty='mcp', lambda_=1e-4,
                 theta=.25, max_iters=10000, tol=1e-3,
                 verbose=True, warm_start=True):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.coef_ = None

    def _get_loss(self, X, y):
        losses = {
            'logistic': Logistic(X, y),
        }
        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            'mcp': MCP(self.lambda_, self.theta),
            'scad': SCAD(self.lambda_, self.theta),
            'lasso': LASSO(self.lambda_, self.theta),
        }
        return penalties[self.penalty]

    def fit(self, X, y):
        # self._set_label_transformers(y)
        # Y = np.asfortranarray(self.label_binarizer_.transform(y),
        #                       dtype=np.float64)
        # print(X)
        # print(Y)
        # return self._fit(X, Y)
        self._set_label_transformers(y)
        return self._fit(X, y)


class GISTRegressor(BaseRegressor, _BaseGIST):
    """
    Estimator for learning linear regressors by GIST
    """
    def __init__(self, loss='squaredloss', penalty='mcp', lambda_=1e-4,
                 theta=.25, max_iters=10000, tol=1e-3,
                 verbose=True, warm_start=True):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.coef_ = None

    def _get_loss(self, X, y):
        losses = {
            'squaredloss': SquaredLoss(X, y),
        }
        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            'mcp': MCP(self.lambda_, self.theta),
            'scad': SCAD(self.lambda_, self.theta),
            'lasso': LASSO(self.lambda_, self.theta),
        }
        return penalties[self.penalty]

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        # Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        # Y = Y.astype(np.float64)
        # return self._fit(X, Y)
        return self._fit(X, y)