import numpy as np
from numpy.linalg import norm


class DCPenalty:
    def __init__(self, lambda_, theta):
        self.lambda_ = lambda_
        self.theta = theta

    def loss(self, x):
        raise NotImplementedError()

    def grad_h(self, x):
        raise NotImplementedError()

    def prox(self, move, alpha):
        raise NotImplementedError()


class SCAD(DCPenalty):
    def loss(self, x):
        assert (self.theta >= 1)
        xabs = np.abs(x)
        l2 = (np.logical_and(self.lambda_ < xabs, xabs <= self.theta *
                             self.lambda_)) * (
                     x ** 2 - 2 * self.lambda_ * xabs +
                     self.lambda_ ** 2) / (2 * (self.theta - 1))

        l3 = (xabs > self.theta * self.lambda_) * (
                self.lambda_ * xabs - (self.theta + 1) * self.lambda_ ** 2 / 2)

        return np.sum(l2 + l3)

    def grad_h(self, x):
        xabs = np.abs(x)
        l1 = np.logical_and(self.lambda_ < xabs, xabs <= self.theta *
                            self.lambda_) * x / (
                            self.theta - 1) - self.lambda_ / \
                            (self.theta - 1) * np.sign(x)
        l2 = (xabs > self.theta * self.lambda_) * self.lambda_ * np.sign(x)
        return l1 + l2

    def prox(self, move, alpha):
        def split_val(x):
            l1 = (np.abs(x) <= self.lambda_) * self.lambda_ * np.abs(x)
            l2 = (np.abs(x) > self.lambda_) * (np.abs(x) <= self.theta *
                self.lambda_) * (2 * self.theta * self.lambda_ * np.abs(x) -
                x ** 2 - self.lambda_ ** 2) / (2 * (self.theta - 1))
            l3 = (np.abs(x) > self.theta * self.lambda_) * (self.theta + 1) *\
                 self.lambda_ ** 2 / 2
            return l1 + l2 + l3
        candidate = [move, np.zeros_like(move), move + self.lambda_ *
                     np.sign(move) / alpha, (alpha * (1 - self.theta) *
                     move + self.theta * self.lambda_ * np.sign(move)) / (
                     alpha * (1 - self.theta) + 1)]
        val = []
        for c in candidate:
            val.append(split_val(c))
        bar_1 = self.theta * self.lambda_
        bar_2 = self.lambda_ + self.lambda_ / alpha
        bar_3 = self.lambda_ / alpha
        flag_1 = (bar_1 > bar_2)
        flag_2 = (bar_1 <= bar_2) * (bar_1 >= bar_3)
        flag_3 = (bar_1 < bar_3)
        l1 = flag_1 * (np.abs(move) <= bar_3) * candidate[1]
        l2 = flag_1 * (np.abs(move) <= bar_2) * (np.abs(move) > bar_3) * \
             candidate[2]
        l3 = flag_1 * (np.abs(move) <= bar_1) * (np.abs(move) > bar_2) * \
             candidate[3]
        l4 = flag_1 * (np.abs(move) > bar_1) * candidate[0]
        l5 = flag_2 * (np.abs(move) <= bar_3) * candidate[1]
        l6 = flag_2 * (np.abs(move) > bar_3) * (np.abs(move) <= bar_1) * \
             candidate[2]
        l7 = flag_2 * (np.abs(move) >= bar_1) * (np.abs(move) <= bar_2) * \
             np.vstack(candidate[2:] + [candidate[0]]).T[range(
            move.size), np.vstack(val[2:] + [val[0]]).T.argmin(axis=1)]
        l8 = flag_2 * (np.abs(move) > bar_2) * candidate[0]
        l9 = flag_3 * (np.abs(move) <= bar_1) * candidate[0]
        l10 = flag_3 * (np.abs(move) > bar_1) * (np.abs(move) <= bar_3) * \
              np.vstack(candidate[:2] + [candidate[3]]).T[range(
              move.size), np.vstack(val[:2] + [val[3]]).T.argmin(axis=1)]
        l11 = flag_3 * (np.abs(move) > bar_3) * (np.abs(move) <= bar_2) * \
              np.vstack(candidate[1:]).T[range(move.size),
              np.vstack(val[1:]).T.argmin(axis=1)]
        l12 = flag_3 * (np.abs(move) > bar_2) * candidate[3]
        return l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12


class MCP(DCPenalty):
    def loss(self, x):
        l1 = (np.abs(x) <= self.theta * self.lambda_) * x ** 2 / (2 * self.theta)
        l2 = (np.abs(x) > self.theta * self.lambda_) * (
                    self.lambda_ * np.abs(x) - 0.5 * self.theta * self.lambda_ ** 2)
        return self.lambda_ * np.linalg.norm(x, 1) - np.sum(l1 + l2)

    def grad_h(self, x):
        return (np.abs(x) <= self.theta * self.lambda_) * x / self.theta + \
               (np.abs(x) > self.theta * self.lambda_) * self.lambda_ * np.sign(x)

    def prox(self, move, alpha):
        def split_val(x):
            l1 = (np.abs(x) <= self.theta * self.lambda_) * (self.lambda_ *
                np.abs(x) - x ** 2 / (2 * self.theta))
            l2 = (np.abs(x) > self.theta * self.lambda_) * self.theta *  \
                 self.lambda_ ** 2 / 2
            return l1 + l2
        candidate = [move, np.zeros_like(move), self.theta * (alpha * move
                 - self.lambda_) / (alpha * self.theta + 1), self.theta *
                     (alpha * move + self.lambda_) / (alpha *
                                                         self.theta + 1)]
        val = []
        for c in candidate:
            val.append(split_val(c))
        flag_1 = (self.theta * alpha <= 1)
        flag_2 = (self.theta * alpha > 1)
        bar_1 = self.theta * self.lambda_
        bar_2 = self.lambda_ / alpha
        l1 = flag_1 * (np.abs(move) > bar_2) * move
        l2 = flag_1 * (move >= bar_1) * (move <= bar_2) * np.vstack(
            candidate[:3]).T[range(move.size), np.vstack(val[:3]).T.argmin(
            axis=1)]
        l3 = flag_1 * (np.abs(move) < bar_1) * np.vstack(
            candidate[1:]).T[range(move.size), np.vstack(val[1:]).T.argmin(
            axis=1)]
        l4 = flag_1 * (move >= -bar_2) * (move <= -bar_1) * np.vstack(
            candidate[:2] + [candidate[3]]).T[range(
            move.size), np.vstack(val[:2] + [val[3]]).T.argmin(axis=1)]
        l5 = flag_2 * (np.abs(move) > bar_1) * move
        l6 = flag_2 * (move <= bar_1) * (move >= bar_2) * candidate[3]
        l7 = flag_2 * (np.abs(move) < bar_2) * np.vstack(
            candidate[1:]).T[range(move.size), np.vstack(val[1:]).T.argmin(
            axis=1)]
        l8 = flag_2 * (move >= - bar_1) * (move <= - bar_2) * candidate[2]
        return l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8


class LASSO(DCPenalty):
    def loss(self, x):
        return self.lambda_ * norm(x, ord=1)

    def grad_h(self, x):
        return 0.

    def prox(self, move, alpha):
        return np.sign(move) * np.maximum(0, np.abs(move) - self.lambda_ /
                                          alpha)