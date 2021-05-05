import numpy as np


class LbfgsInvHessProduct(object):
    """Linear operator for the L-BFGS approximate inverse Hessian.
    This operator computes the product of a vector with the approximate
    inverse of the Hessian of the objective function, using the L-BFGS
    limited memory approximation to the inverse Hessian, accumulated during
    the optimization.

    Parameters
    ----------
    sk : array_like, shape=(n_corr, n)
    yk : array_like, shape=(n_corr, n)
    """
    def __init__(self, sk, yk):
        """Construct the operator."""
        if sk.shape != yk.shape or sk.ndim != 2:
            raise ValueError('sk and yk must have matching shape, (n_corrs, n)')
        n_corrs, n = sk.shape

        self.sk = sk
        self.yk = yk
        self.n_corrs = n_corrs
        self.rho = 1 / np.einsum('ij, ij->i', sk, yk)

    def _matvec(self, x):
        """Efficient matrix-vector multiply with the BFGS matrices.

        Parameters
        ----------
        x : ndarray
            An array with shape (n,) or (n, 1).

        Returns
        -------
        y : ndarray
            The matrix-vector product
        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        q = np.array(x, copy=True)
        if q.ndim == 2 and q.shape[1] == 1:
            q = q.reshape(-1)

        alpha = np.empty(n_corrs)

        for i in range(n_corrs-1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q = q - alpha[i]*y[i]
        r = q
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)

        return r

    def todense(self):
        """Return a dense array representation of this opertor.

        Returns
        -------
        arr : ndarray, shape=(n,n)

        """
        s, y, n_corrs, rho = self.sk, self.yk, self.n_corrs, self.rho
        I = np.eye(s.shape[1])
        Hk = I

        for i in range(n_corrs):
            A1 = I - s[i][:, np.newaxis] * y[i][np.newaxis, :] * rho[i]
            A2 = I - y[i][:, np.newaxis] * s[i][np.newaxis, :] * rho[i]

            Hk = np.dot(A1, np.dot(Hk, A2)) + (rho[i] * s[i][:, np.newaxis]
                                                * s[i][np.newaxis, :])
        return Hk