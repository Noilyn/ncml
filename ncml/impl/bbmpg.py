from .loss import Logistic, SquaredLoss
from .penalty import SCAD, MCP, LASSO
from .base import BaseClassifier, BaseRegressor
import numpy as np
import time
from numpy.linalg import norm


class _BaseBBMPG(object):
    """
    Variable Metric Proximal Gradient Method with Diagonal Brazilai-Browein
    stepsize
    Solve the following problem:
        minimize f(x) = l(x) + r(x)
        f(x): bounded below
        l(x): continuously differentiable with Lipschitz continuous gradient
        r(x): can be non-smooth and non-convex, can be rewritten as the
        difference of two convex function, i.e., r(x) = r_1(x) - r_2(x)
    """
    def _get_loss(self, X, y):
        return NotImplementedError()

    def _get_penalty(self):
        raise NotImplementedError()

    def _fit(self, X, y):
        loss = self._get_loss(X, y)
        penalty = self._get_penalty()
        n_samples, n_features = X.shape
        l_const = loss.l_const

        # initialization
        if self.warm_start:
            if self.coef_ is None:
                w_init = np.zeros(n_features)
            else:
                w_init = self.coef_

        w_old = w_init.copy()
        w = w_init.copy()
        self.obj_pass = [loss.loss(w_old) + penalty.loss(w_old)]
        self.execution_time_pass = [0]
        t_start = time.time()
        scale = l_const

        for k in range(self.max_iters):
            try:
                if k:
                    # compute standard bb-step
                    sub_s = w - w_old
                    sub_y = loss.grad(w) - loss.grad(w_old)
                    if self.scale_choice == 'bb_1':
                        scale = sub_s.dot(sub_s) / sub_s.dot(sub_y)
                    elif self.scale_choice == 'bb_2':
                        scale = sub_s.dot(sub_y) / sub_y.dot(sub_y)
                    elif self.scale_choice == 'diagonal_bb':
                        bb_1 = sub_s.dot(sub_s) / sub_s.dot(sub_y)
                        bb_2 = sub_s.dot(sub_y) / sub_y.dot(sub_y)
                        if not k:
                            bb_diagonal_old = 1 / bb_2
                        diagonal_ini = (sub_s * sub_y + self.mu *
                                        bb_diagonal_old) / (sub_s * sub_s + self.mu)
                        bar_1, bar_2 = 1 / bb_1, 1 / bb_2
                        scale = bar_1 * (diagonal_ini < bar_1) + bar_2 * (
                            diagonal_ini > bar_2) + diagonal_ini * (diagonal_ini
                            >= bar_1) * (diagonal_ini <= bar_2)
                    w_old = w
                while True:
                    move = w - loss.grad(w) / scale
                    w_new = penalty.prox(move, scale)
                    loss_for_criterion = loss.loss(w_new) + penalty.loss(w_new)
                    sub_for_criterion = w_new - w
                    sub_val_for_criterion = (sub_for_criterion *
                                             scale).dot(sub_for_criterion)/2
                    if self.linesearch_choice == 'nonmonotonic':
                        if loss_for_criterion < max(self.obj_pass[max(0, k -
                                self.m_ls + 1):(k + 1)]) - sub_val_for_criterion:
                            w = w_new
                            if self.scale_choice == 'diagonal_bb':
                                bb_diagonal_old = scale
                            break
                    elif self.linesearch_choice == 'monotonic':
                        loss_before = loss.loss(w) + penalty.loss(w)
                        if loss_for_criterion <= loss_before:
                            w = w_new
                            if self.scale_choice == 'diagonal_bb':
                                bb_diagonal_old = scale
                            break
                    scale *= self.beta
            except:
                break
            # profile

            rel = norm(w_old - w) / (1 + norm(w))
            execution_cost = time.time() - t_start
            self.obj_pass.append(loss_for_criterion)
            self.execution_time_pass.append(execution_cost)
            if self.verbose and (k % 10 == 0 or k == self.max_iters - 1):
                print('Iter: %d, obj: %.3e, nnz: %.3e, rel: %.3e, execution '
                      'time: %.3e' % (k, loss_for_criterion, sum(w != 0),
                                      rel, execution_cost))
            if rel < self.tol or k == (self.max_iters-1):
                self.iter_num = k
                self.val = loss_for_criterion
                self.nnz = sum(w != 0)
                self.execution_time_cost = execution_cost
                break
        self.coef_ = w


class BBMPGClassifier(BaseClassifier, _BaseBBMPG):
    """
            Estimator for learning linear classifiers by BBMPG
    """
    def __init__(self, loss='logistic', penalty='mcp', lambda_=1e-4, theta=.25,
                m_ls=15, beta=2, mu=0.01, max_iters=10000, tol=1e-3,
                 scale_choice='diagonal_bb',
                 linesearch_choice='nonmonotonic',
                 verbose=True, warm_start=True):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.m_ls = m_ls
        self.beta = beta
        self.mu = mu
        self.max_iters = max_iters
        self.tol = tol
        self.scale_choice = scale_choice
        self.linesearch_choice = linesearch_choice
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


class BBMPGRegressor(BaseRegressor, _BaseBBMPG):
    """
        Estimator for learning linear regressors by BBMPG
    """
    def __init__(self, loss='squaredloss', penalty='mcp', lambda_=1e-4,
                 theta=.25, m_ls=15, beta=2, mu=0.01, max_iters=10000, tol=1e-3,
                 scale_choice='diagonal_bb',
                 linesearch_choice='nonmonotonic',
                 verbose=True, warm_start=True):
        self.loss = loss
        self.penalty = penalty
        self.lambda_ = lambda_
        self.theta = theta
        self.m_ls = m_ls
        self.beta = beta
        self.mu = mu
        self.max_iters = max_iters
        self.tol = tol
        self.scale_choice = scale_choice
        self.linesearch_choice = linesearch_choice
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