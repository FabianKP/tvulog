
from matplotlib import pyplot as plt
import numpy as np
from skimage.data import camera


from src.tvulog.differential_operators._forward_difference import ForwardDifference, gradient_norms, forward_difference


RTOL = 1e-6     # Relative tolerance for numerical errors.
PLOT = False


def test_variation_simple():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    f = x**2
    dfdx_true = [3, 5, 7, 9, 0]
    dx = ForwardDifference(shape=f.shape, width_to_height=1.)
    dfdx = dx.matvec(f)
    assert np.isclose(dfdx, dfdx_true).all()


def test_adjoint1d():
    n = 8
    v = ForwardDifference(shape=(n,), width_to_height=1.)
    # Assemble gradient matrix.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(n)]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([v.rmatvec(e_i) for e_i in np.eye(n)]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()


def test_adjoint2d():
    m = 10
    n = 12
    v = ForwardDifference(shape=(m, n), width_to_height=1.)
    # Assemble gradient matrix.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(v.shape[1])]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([v.rmatvec(e_i) for e_i in np.eye(v.shape[0])]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()


def test_on_image():
    img = camera().astype(float)
    m, n = img.shape
    nabla = ForwardDifference(shape=(m, n), width_to_height=1.)
    diff1, diff2 = nabla.fwd_arr(img)
    if PLOT:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        ax[1].imshow(diff1.reshape(m, n))
        ax[2].imshow(diff2.reshape(m, n))
        plt.show()


def test_forward_difference_function():
    m = 12
    n = 53
    wth = 2.4
    np.random.seed(42)
    test_img = np.random.randn(m, n)
    nabla = ForwardDifference(shape=(m, n), width_to_height=wth)
    nabla_f = nabla @ test_img.flatten()
    f1 = forward_difference(test_img, width_to_height=wth, axis=0).flatten()
    f2 = forward_difference(test_img, width_to_height=wth, axis=1).flatten()
    diff_f = np.concatenate([f1, f2])
    assert np.isclose(nabla_f, diff_f).all()


def test_gradient_norms():
    img = camera().astype(float)
    gradnorms_img = gradient_norms(img)
    assert gradnorms_img.shape == img.shape
    assert np.all(gradnorms_img >= 0.)
    if PLOT:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(gradnorms_img)
        plt.show()


def test_adjoint3d():
    k = 5
    m = 10
    n = 12
    v = ForwardDifference(shape=(k, m, n), width_to_height=1.)
    # Assemble gradient matrix.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(v.shape[1])]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([v.rmatvec(e_i) for e_i in np.eye(v.shape[0])]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()


def test_csr1d():
    n = 8
    v = ForwardDifference(shape=(n,), width_to_height=1.)
    # Assemble matrix representation.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(n)]).T
    # Get csr-matrix.
    csr_mat = v.csr_matrix
    csr_dense = csr_mat.todense()
    assert np.isclose(mat, csr_dense).all()


def test_csr2d():
    m = 10
    n = 12
    v = ForwardDifference(shape=(m, n), width_to_height=1.)
    # Assemble matrix representation.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(v.shape[1])]).T
    # Get csr-matrix.
    csr_dense = v.csr_matrix.toarray()
    assert np.isclose(mat, csr_dense).all()


def test_csr3d():
    k = 3
    m = 3
    n = 3
    v = ForwardDifference(shape=(k, m, n), width_to_height=1.)
    # Assemble matrix representation.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(v.shape[1])]).T
    # Get csr-matrix.
    csr_dense = v.csr_matrix.toarray()
    if PLOT:
        fig, ax = plt.subplots(1, 2)
        ax[0].spy(mat)
        ax[1].spy(csr_dense)
        plt.show()
    assert np.isclose(mat, csr_dense).all()