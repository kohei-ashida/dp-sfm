# %%
import numpy as np
import scipy.sparse
from tqdm import tqdm
import scipy.sparse
from tqdm import tqdm

try:
    import cupy as cp
    import cupyx.scipy as cpx
except:
    pass


def L1_residual_min_cupy(A, b, max_ite=1000, tol=1.0e-8):
    print("use cupy")

    # cpのattributeを表示
    # print(f"cp.cuda.Device() = {cp.cuda.Device()}")

    A = cp.asarray(A)
    b = cp.asarray(b)
    print("convert to cupy array, shape = ", A.shape)

    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)
        minimize ||Ax - b||_1
    :param A: A design matrix (Cupy 2D array)
    :param b: A column vector as a Cupy 2D array
    :param max_ite: Maximum number of iterations
    :param tol: Tolerance
    :return: An approximate solution `x` that minimizes ||Ax - b||_0.
    Raises:
        ValueError: An error occurs in evaluating the dimensionality of the input matrix A and vector b.
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    eps = 1.0e-8
    m, n = A.shape

    xold = cp.ones((n, 1))
    W = cpx.sparse.identity(m).tocsc()
    if cp.ndim(b) != 2 or b.shape[1] != 1:
        raise ValueError("b needs to be a column vector m x 1")

    iter = 0
    with tqdm(total=max_ite) as pbar:
        while iter < max_ite:
            pbar.update(1)
            iter = iter + 1
            # Solve the weighted least squares WAx=Wb
            x = cp.linalg.lstsq(W.dot(A), W.dot(b), rcond=None)[0]
            r = b - A.dot(x)
            # Termination criterion
            if cp.linalg.norm(x - xold) < tol:
                print("satisfied tol")
                x_np = cp.asnumpy(x)
                # cp.cuda.set_allocator()
                return x_np
            else:
                xold = x
            # Update weighting factor
            W = cpx.sparse.diags(
                1.0 / cp.maximum(cp.sqrt(cp.fabs(r)), eps).T[0]
            ).tocsc()
    x_np = cp.asnumpy(x)
    # cp.cuda.set_allocator()
    return x_np


def L1_residual_min(A, b, max_ite=1000, tol=1.0e-8):
    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)
        minimize ||Ax - b||_1
    :param A: A design matrix (numpy 2D array)
    :param b: A column vector as a numpy 2D array
    :param max_ite:Maximum number of iterations
    :param tol: Tolerance
    :return: An approximate solution `x` that minimizes ||Ax - b||_0.
    Raises:
        ValueError: An error occurs in evaluating the dimensionality of the input matrix A and vector b.
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    eps = 1.0e-8
    m, n = A.shape

    xold = np.ones((n, 1))
    # W = np.identity(m)を疎行列を用いて表現
    W = scipy.sparse.identity(m)
    if np.ndim(b) != 2 and b.shape[1] != 1:
        raise ValueError("b needs to be a column vector m x 1")

    iter = 0
    with tqdm(total=max_ite) as pbar:
        while iter < max_ite:
            pbar.update(1)
            iter = iter + 1
            # Solve the weighted least squares WAx=Wb
            x = np.linalg.lstsq(W.dot(A), W.dot(b), rcond=None)[0]
            r = b - A.dot(x)
            # Termination criterion
            if np.linalg.norm(x - xold) < tol:
                print("satisfied tol")
                return x
            else:
                xold = x
            # Update weighting factor
            # W = np.diag(np.asarray(
            #     1.0 / np.maximum(np.sqrt(np.fabs(r)), eps))[:, 0])を疎行列で
            W = scipy.sparse.diags(1.0 / np.maximum(np.sqrt(np.fabs(r)), eps).T[0])
        return x


def kinji_L1(x, y):
    x_ = x.reshape(-1, 1)
    x_ = np.hstack((x_, np.ones_like(x_)))

    y_ = y.reshape(-1, 1)

    param = L1_residual_min_cupy(x_, y_)

    A, B = *param[0], *param[1]
    return A, B