from ncml.impl.loss import Logistic, SquaredLoss
from ncml.impl.penalty import SCAD, MCP, LASSO
from ncml.impl.base import BaseClassifier, BaseRegressor
import numpy as np
from numpy.linalg import norm
import time


class _BaseDCA:
    """
    Proximal difference-of-convex algorithm with extrapolation

    Solve the following problem:
        minimize F(x) := f(x) + P(x)
        F(x): bounded below
        f(x): smooth convex function with Lipschitz continuous gradient whose
        modulus is L > 0
        P(x): P(x) = P_1(x) - P_2(x) with P_1 being a proper closed convex
        function and P_2 being a continuous convex function
    """
    def _get_loss(self, X, y):
        raise NotImplementedError()

    def _get_penalty(self):
        raise NotImplementedError()

    def _fit(self, X, y, w_init=None):
        loss = self._get_loss(X, y)
        penalty = self._get_penalty()
        n_samples, n_features = X.shape
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
        if self.penalty in ['mcp', 'scad']:
            prox_type = 'scaled_l1'
        if prox_type == 'scaled_l1':
            prox_opt = LASSO(self.lambda_, self.theta)

        for k in range(self.max_iters):
            try:
                if not k:
                    # profile
                    self.obj_pass = [loss.loss(w_old) + penalty.loss(w_old)]
                    self.execution_time_pass = [0]
                    t_start = time.time()
                    # initialization
                    if self.momentum_flag:
                        # represent theta in original paper
                        tta_before = 1
                        tta = 1
                        momentum = w
                if self.momentum_flag:
                    beta = (tta_before - 1) / tta
                    tta_before = tta
                    tta = (1 + np.sqrt(1 + 4 * tta_before ** 2)) / 2
                    momentum_old = momentum
                    momentum = w + beta * (w - w_old)
                    if self.restart_scheme == 'adaptive':
                        if np.dot(momentum_old-w, w-w_old) > 0:
                            tta = 1
                            tta_before = 1
                    elif self.restart_scheme == 'fixed':
                        if (k+1) == self.restart_epoch:
                            tta = 1
                            tta_before = 1
                else:
                    momentum = w

                move = momentum - (loss.grad(momentum) - penalty.grad_h(w)) /\
                       loss.l_const
                w_old = w

                w = prox_opt.prox(move, loss.l_const)

                # profile
                rel = norm(w_old - w) / (1 + norm(w))
                tcost = time.time() - t_start
                self.obj_pass.append(loss.loss(w) + penalty.loss(w))
                self.execution_time_pass.append(tcost)
                if self.verbose and (k % 10 == 0 or k == self.max_iters - 1):
                    print(
                        'Iter %d, obj %.3e, nnz %.3e, rel %.3e, time %.3e' %
                        (k, self.obj_pass[-1], sum(w != 0), rel, tcost)
                    )
            except:
                break
            if rel < self.tol or k == (self.max_iters-1):
                self.iter_num = k
                self.val = self.obj_pass[-1]
                self.nnz = sum(w != 0)
                self.execution_time_cost = tcost
                break
        self.coef_ = w


class DCAClassifier(BaseClassifier, _BaseDCA):
    """
        Estimator for learning linear classifiers by pdeca
    """
    # pycharm 自动转doc
    # restart_scheme can be 'adaptive' or 'fixed'
    # restart_epoch worked only if restart_scheme == 'fixed'
    def __init__(self, loss='logistic', penalty='mcp', lambda_=1e-4,
                 theta=.25, max_iters=10000, tol=1e-4, restart_epoch=500,
                 restart_scheme='adaptive', momentum_flag=True, verbose=True,\
                 warm_start=False):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.max_iters = max_iters
        self.tol = tol
        self.restart_epoch = restart_epoch
        self.restart_scheme = restart_scheme
        self.momentum_flag = momentum_flag
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


class DCARegressor(BaseRegressor, _BaseDCA):
    """
        Estimator for learning linear classifiers by pdeca
    """

    # TODO 参数含义要写doc
    # pycharm 自动转doc
    # restart_scheme can be 'adaptive' or 'fixed'
    # restart_epoch worked only if restart_scheme == 'fixed'
    def __init__(self, loss='squaredloss', penalty='mcp', lambda_=1e-4,
                 theta=.25, max_iters=10000, tol=1e-4, restart_epoch=500,
                 restart_scheme='adaptive', momentum_flag=True, verbose=True,\
                 warm_start=False):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.max_iters = max_iters
        self.tol = tol
        self.restart_epoch = restart_epoch
        self.restart_scheme = restart_scheme
        self.momentum_flag = momentum_flag
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
        }
        return penalties[self.penalty]

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        # Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        # Y = Y.astype(np.float64)
        # return self._fit(X, Y)
        return self._fit(X, y)
