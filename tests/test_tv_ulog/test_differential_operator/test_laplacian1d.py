
import numpy as np
from scipy.ndimage import laplace
from time import time

from src.tvulog.differential_operators._laplacian1d import Laplacian1D


RTOL = 1e-6     # Relative tolerance for numerical errors.


def test_laplacian_simple():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    f = x**2
    dfdx_true = 2 * np.ones_like(x)
    dx = Laplacian1D(size=f.size)
    dfdx = dx.matvec(f)
    assert np.isclose(dfdx[1:-1], dfdx_true[1:-1]).all()


def test_matrix_shape():
    """
    Check that the one-dimensional Laplacian really has the correct shape.
    """
    n = 5
    lap = Laplacian1D(size=n)
    lap_mat = lap.csr_matrix.toarray()
    target = np.array([[-2, 2, 0, 0, 0],
                       [1, -2, 1, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 1, -2, 1],
                       [0, 0, 0, 2, -2]])
    assert np.isclose(lap_mat, target).all()


def test_what_is_faster():
    # What is faster, applying Laplacian via convolution or as sparse matrix.
    num_tests = 50
    n = 100000
    delta = Laplacian1D(size=n)
    x = np.random.randn(num_tests, n)
    t0 = time()
    for xi in x:
        x1 = laplace(xi, mode="mirror")
    t1 = time() - t0
    t0 = time()
    for xi in x:
        x2 = delta.matvec(xi)
    t2 = time() - t0
    print(f"Time of convolution: {t1}.")
    print(f"Time of sparse multiplication: {t2}")


def test_what_is_faster_adjoint():
    # What is faster, applying Laplacian via convolution or as sparse matrix.
    num_tests = 50
    n = 100000
    delta = Laplacian1D(size=n)
    x = np.random.randn(num_tests, n)
    t0 = time()
    for xi in x:
        x1 = laplace(xi, mode="constant")
    t1 = time() - t0
    t0 = time()
    for xi in x:
        x2 = delta.rmatvec(xi)
    t2 = time() - t0
    print(f"Time of convolution: {t1}.")
    print(f"Time of sparse multiplication: {t2}")


def test_adjoint1d():
    n = 8
    delta = Laplacian1D(size=n)
    # Assemble gradient matrix.
    mat = np.array([delta.matvec(e_i) for e_i in np.eye(n)]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([delta.rmatvec(e_i) for e_i in np.eye(n)]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()


def test_csr1d():
    n = 8
    delta = Laplacian1D(size=n)
    # Assemble gradient matrix.
    mat = np.array([delta.matvec(e_i) for e_i in np.eye(n)]).T
    csr_mat = delta.csr_matrix
    csr_dense = csr_mat.toarray()
    # Compare
    assert np.isclose(mat, csr_dense).all()