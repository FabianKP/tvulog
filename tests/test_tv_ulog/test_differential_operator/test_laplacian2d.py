
from matplotlib import pyplot as plt
import numpy as np
from skimage.data import camera
from scipy.ndimage import laplace, generic_laplace, correlate1d
from time import time

from src.tvulog.differential_operators._laplacian2d import Laplacian2D


RTOL = 1e-6     # Relative tolerance for numerical errors.


def test_laplacian2d_init():
    m = 12
    n = 53
    laplacian_op = Laplacian2D(m=m, n=n, width_to_height=1)
    assert laplacian_op.x_size == m * n
    assert laplacian_op.shape == (m * n, m * n)


def test_look_at_matrix():
    m = 4
    n = 4
    delta = Laplacian2D(m=m, n=n, width_to_height=1)
    delta_mat = delta.csr_matrix.toarray()
    assert delta_mat.shape == (m * n, m * n)
    print(" ")
    print(delta_mat)


def test_agrees_with_convolution_in_mirror_mode():
    np.random.seed(42)
    num_tests = 10
    m = 10
    n = 12
    for _ in range(num_tests):
        img = np.random.randn(m, n)
        x = img.flatten()
        delta = Laplacian2D(m, n, width_to_height=1)
        delta_x = delta.matvec(x)
        delta_img = delta_x.reshape(m, n)
        convolved_img = laplace(img, mode="mirror")
        assert np.isclose(delta_img, convolved_img).all()


def test_what_is_faster():
    """
    What is faster, applying Laplacian via convolution or as sparse matrix?
    (This is not a proper test)
    """
    m = 123
    n = 100
    num_tests = 50
    np.random.seed(42)
    delta = Laplacian2D(m=m, n=n, width_to_height=1)
    x = np.random.randn(num_tests, m, n)
    t0 = time()
    for xi in x:
        x1 = laplace(xi, mode="mirror")
    t1 = time() - t0
    t0 = time()
    for xi in x:
        x2 = delta.matvec(xi.flatten())
    t2 = time() - t0
    print("\n")
    print(f"Time of convolution: {t1}.")
    print(f"Time of sparse multiplication: {t2}")


def test_what_is_faster_adjoint():
    # What is faster, applying Laplacian via convolution or as sparse matrix.
    num_tests = 50
    m = 123
    n = 100
    delta = Laplacian2D(m=m, n=n, width_to_height=1)
    x = np.random.randn(num_tests, m, n)
    t0 = time()
    for xi in x:
        x1 = laplace(xi, mode="constant")
    t1 = time() - t0
    t0 = time()
    for xi in x:
        x2 = delta.rmatvec(xi.flatten())
    t2 = time() - t0
    print("\n")
    print(f"Time of convolution: {t1}.")
    print(f"Time of sparse multiplication: {t2}")


def test_on_large_image():
    img = camera()
    # Convert image to float.
    img = img.astype(float)
    m, n = img.shape
    delta = Laplacian2D(m, n, width_to_height=1)
    x = img.flatten()
    delta_x = delta.matvec(x)
    delta_img = delta_x.reshape(m, n)
    convolved_img = laplace(img, mode="mirror")
    assert np.isclose(delta_img, convolved_img).all()


def test_adjoint():
    m = 10
    n = 12
    delta = Laplacian2D(m, n, width_to_height=1)
    # Assemble gradient matrix.
    mat = np.array([delta.matvec(e_i) for e_i in np.eye(m*n)]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([delta.rmatvec(e_i) for e_i in np.eye(m*n)]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()
    assert np.isclose(mat.T, adj_mat).all()


def test_csr():
    m = 12
    n = 52
    delta = Laplacian2D(m, n, width_to_height=1.)
    # Assemble Laplace matrix.
    mat = np.array([delta.matvec(e_i) for e_i in np.eye(m * n)]).T
    csr_mat = delta.csr_matrix
    csr_dense = csr_mat.toarray()
    # Compare
    assert np.isclose(mat, csr_dense).all()


def test_ratio():
    """
    Check if the `width_to_height`-parameter has the desired consequences.
    """
    np.random.seed(42)  # as always, want a deterministic test behavior
    m = 12
    n = 53
    test_img = np.random.randn(m, n)
    wth_list = [2, 0.5, 1]
    for width_to_height in wth_list:
        delta = Laplacian2D(m=m, n=n, width_to_height=width_to_height)
        weights = np.array([1., 1 / (width_to_height ** 2)])
        def derivative_scaled(_image, _axis, _output, _mode, _cval):
            return weights[_axis] * correlate1d(_image, [1, -2, 1], _axis, _output, _mode, _cval, 0)

        laplace_img = generic_laplace(test_img, derivative_scaled, mode="mirror")
        x = test_img.flatten()
        delta_x = delta.matvec(x)
        delta_img = delta_x.reshape(m, n)
        assert np.isclose(delta_img, laplace_img).all()