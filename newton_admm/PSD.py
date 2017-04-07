import numpy as np
import math


def vec(matin): return matin.T.ravel()


def mat(vecin, nr, nc): return np.reshape(vecin, (nc, nr)).T


_sqrt2 = math.sqrt(2)


def X_to_vec(X):
    n = X.shape[0]
    return X.T[np.tri(n, dtype=np.bool).T]


def vec_to_X(v_X):
    n = int(math.sqrt(2 * len(v_X)))
    if len(v_X) != n * (n + 1) / 2:
        raise ValueError(
            "v_X is not the right shape for a vectorized lower triangular matrix. Tried to turn vector of size {} into matrix with width {} ".format(len(v_X), n))
    Y = np.zeros((n, n))
    Y[np.tri(n, dtype=np.bool).T] = v_X
    return Y + np.triu(Y, 1).T


def dot(A, B, axes=(1, 0)):
    # I should document what happens to the order of the remaining axes
    if not isinstance(axes, tuple) or len(axes) != 2:
        raise ValueError("Incorrect axes parameter.")
    if A.shape[axes[0]] != B.shape[axes[1]]:
        raise ValueError("Dimension mismatch")
    rolled_A = np.rollaxis(A.transpose(), A.ndim - 1 - axes[0]).transpose()
    rolled_B = np.rollaxis(B.transpose(), B.ndim - 1 - axes[1], 1).transpose()
    return rolled_A.dot(rolled_B)


def multiply_diag(A, D):
    # multiply a matrix A by diagonal tensor D
    n1, n2 = A.shape
    n3, n4, n5 = D.shape
    assert(n2 == n3)
    out = np.zeros((n1, n2, n4, n5))
    for i in range(n4):
        for j in range(n5):
            out[:, :, i, j] = A * D[:, i, j]
    return out


def P(v_X):
    X = vec_to_X(v_X)
    n = X.shape[0]
    # perform scaling
    _i = tuple(range(n))
    X[_i, _i] *= _sqrt2

    Lam, U = np.linalg.eigh(X)
    idx = np.argsort(Lam)
    Lam = Lam[idx]
    U = U[:, idx]
    Lam_max = np.maximum(Lam, 0.0)
    out = U.dot(np.diag(Lam_max).dot(U.T))
    # undo scaling
    out[_i, _i] /= _sqrt2
    return X_to_vec(out)


def dU_dL_dX(Lam, U):
    # compute dU_dX and dL_dX given U and L
    n = len(Lam)
    _i = tuple(range(n))
    idx = np.argsort(Lam)
    Lam = Lam[idx]
    U = U[:, idx]
    dU_dX = np.zeros((n, n, n, n))
    for i, (l0, u0) in enumerate(zip(Lam, U.T)):
        d = (l0 - Lam)
        d[d != 0] = 1. / d[d != 0]
        inv = (U.dot(np.diag(d)).dot(U.T))
        tmp = np.multiply.outer(inv, u0)
        tmp += np.rollaxis(tmp, 2, 1)
        tmp[:, _i, _i] /= 2.0
        # for j in range(n):
        #     tmp[:,j,j] /= 2.0
        # set the ith column
        dU_dX[:, i, :, :] = tmp

    dL_dX = np.zeros((n, n, n))
    for i, u0 in enumerate(U.T):
        tmp = 2 * np.multiply.outer(u0, u0)
        for j in range(n):
            tmp[j, j] /= 2.0
        dL_dX[i, :, :] = tmp
    return dU_dX, dL_dX


def J(v_X):
    X = vec_to_X(v_X)
    n = X.shape[0]

    # perform scaling
    _i = tuple(range(n))
    X[_i, _i] *= _sqrt2

    Lam, U = np.linalg.eigh(X)
    idx = np.argsort(Lam)
    Lam = Lam[idx]
    U = U[:, idx]
    L = np.diag(Lam)
    L_max = np.maximum(L, 0.0)

    dU_dX, dL_dX = dU_dL_dX(Lam, U)
    dL_max_dX = dL_dX.copy()
    for i, l in enumerate(Lam):
        if l < 0:
            dL_max_dX[i, :, :] = 0
    t1 = dot(U.dot(L_max), np.rollaxis(dU_dX, 1, 0))
    t2 = np.rollaxis(t1, 1, 0)
    t3 = np.rollaxis(dot(multiply_diag(U, dL_max_dX), U.T, (1, 0)), 3, 1)

    idx = np.nonzero(np.tri(n, dtype=np.bool).T)
    W = t1 + t2 + t3

    # rescale jacobian
    W[:, :, _i, _i] *= _sqrt2
    W[_i, _i, :, :] /= _sqrt2

    return W[idx[0], idx[1]][:, idx[0], idx[1]]
