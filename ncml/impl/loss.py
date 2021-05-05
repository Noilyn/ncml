import numpy as np
from scipy.sparse import csr_matrix, spdiags
from sklearn.utils import shuffle
from .untils import _mat2norm


class LossFunction(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def loss(self, w):
        raise NotImplementedError()

    def grad(self, w):
        raise NotImplementedError()


class Logistic(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)
        bA = spdiags(y, 0, len(y), len(y)) * X
        self.l_const = _mat2norm(bA) ** 2 / self.X.shape[0]

    def loss(self, w):
        z = -self.y * self.X.dot(w)
        l = np.zeros(z.shape[0])
        l[z >= 0] = z[z >= 0] + np.log(1 + np.exp(-z[z >= 0]))
        l[z < 0] = np.log(1 + np.exp(z[z < 0]))
        return np.mean(l)

    def grad(self, w):
        z = -self.y * self.X.dot(w)
        p = np.zeros(z.shape[0])
        p[z < 0] = np.exp(z[z < 0]) / (1 + np.exp(z[z < 0]))
        p[z >= 0] = 1 / (1 + np.exp(-z[z >= 0]))
        return self.X.T.dot(-self.y * p) / self.X.shape[0]

    def batch_grad(self, w, batch_size):
        X_sub, y_sub = shuffle(self.X, self.y, n_samples=batch_size)
        z = -y_sub * X_sub.dot(w)
        p = np.zeros(z.shape[0])
        p[z < 0] = np.exp(z[z < 0]) / (1 + np.exp(z[z < 0]))
        p[z >= 0] = 1 / (1 + np.exp(-z[z >= 0]))
        return X_sub.T.dot(-y_sub * p) / X_sub.shape[0]


class SquaredLoss(LossFunction):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.l_const = _mat2norm(X) ** 2 / self.X.shape[0]

    def loss(self, w):
        n_samples, n_shapes = self.X.shape
        diff = self.y - self.X.dot(w)
        return diff.dot(diff) / (2 * n_samples)

    def grad(self, w):
        n_samples, n_shapes = self.X.shape
        return csr_matrix.transpose(self.X).dot(self.X.dot(w) - self.y) / \
               n_samples

    def batch_grad(self, w, batch_size):
        X_sub, y_sub = shuffle(self.X, self.y, n_samples=batch_size)
        n_samples, n_shapes = self.sub_X.shape
        return csr_matrix.transpose(X_sub).dot(X_sub.dot(w) - y_sub) / n_samples
