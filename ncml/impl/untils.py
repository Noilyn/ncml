import math
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix


def _mat2norm(A, eps=1e-3):
    """
    Compute matrix 2 norm by power method
    """
    [n, d] = A.shape
    x = np.random.normal(size=d)
    x /= norm(x)
    x_old = x
    for i in range(100):
        y = A.T.dot(A.dot(x_old))
        x = y / norm(y)
        if abs(x.dot(x_old)) > 1 - eps:
            break
        x_old = x
    lamda = math.sqrt(x.dot(A.T.dot(A.dot(x))))
    return lamda


def reg_synthetic_reproduce(n_samples, n_features, nnz_num, csr_flag=False):
    X = np.random.randn(n_samples, n_features)
    w = np.zeros(n_features)
    transformer = MaxAbsScaler().fit(X)
    X = transformer.transform(X)
    idx = np.random.choice(np.arange(n_features), nnz_num)
    w[idx] = np.random.randn(nnz_num)
    y = X.dot(w) + 0.01 * np.random.randn(n_samples)
    if csr_flag:
        X = csr_matrix(X)
    return X, y