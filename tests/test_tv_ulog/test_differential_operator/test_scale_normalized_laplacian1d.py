
import numpy as np

from src.tvulog.differential_operators._scale_normalized_laplacian1d import ScaleNormalizedLaplacian1D
from src.tvulog.differential_operators._laplacian1d import Laplacian1D
from src.tvulog.differential_operators._scale_normalized_forward_difference1d import ScaleNormalizedForwardDifference1D


np.random.seed(42)


def test_construct():
    n = 12
    sigmas = [1, 2, 3]
    k = len(sigmas)
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=sigmas)
    delta_norm_mat = delta_norm.csr_matrix
    assert delta_norm.shape == (k * n, k * n)
    assert delta_norm.x_size == k * n
    assert delta_norm_mat.shape == (k * n, k * n)


def test_print_csr_mat():
    n = 5
    sigmas = [1, 2, 3]
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=sigmas)
    csr_mat = delta_norm.csr_matrix
    print("\n")
    print(csr_mat.toarray())


def test_fwd_action():
    """
    Tests on random vectors if the forward action of the scale-normalized Laplacian is correct.
    """
    np.random.seed(42)  # makes test deterministic
    num_tests = 10
    n = 123
    k = 23
    sigmas = np.arange(1, 1 + k)
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=sigmas)
    delta = Laplacian1D(size=n)
    scales = np.square(sigmas)
    for _ in range(num_tests):
        x = np.random.randn(k, n)
        # Compute scale-normalized Laplacian manually
        delta_norm_x_ref = [t * delta.matvec(x_i) for t, x_i in zip(scales, x)]
        delta_norm_x_ref = np.array(delta_norm_x_ref)
        # Now, compute with `matvec`.
        delta_norm_x = delta_norm.matvec(x.flatten())
        delta_norm_x = delta_norm_x.reshape(k, n)
        assert np.isclose(delta_norm_x, delta_norm_x_ref).all()


def test_adj_action():
    """
    Tests on random vectors if the adjoint action of the scale-normalized Laplacian is correct.
    Note that the adjoint of the scale-normalized Laplacian is simply the scale-normalized adjoint Laplacian!
    """
    np.random.seed(42)  # makes test deterministic
    num_tests = 10
    n = 123
    k = 23
    sigmas = np.arange(1, 1 + k)
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=sigmas)
    delta = Laplacian1D(size=n)
    scales = np.square(sigmas)
    for _ in range(num_tests):
        x = np.random.randn(k, n)
        # Compute adjoint scale-normalized Laplacian manually
        adj_delta_norm_x_ref = [t * delta.rmatvec(x_i) for t, x_i in zip(scales, x)]
        adj_delta_norm_x_ref = np.array(adj_delta_norm_x_ref)
        # Now, compute with `matvec`.
        adj_delta_norm_x = delta_norm.rmatvec(x.flatten())
        adj_delta_norm_x = adj_delta_norm_x.reshape(k, n)
        assert np.isclose(adj_delta_norm_x, adj_delta_norm_x_ref).all()


def test_adjoint_matrix():
    """
    Check that `self.rmatvec(x)` truly corresponds to `self.csr_matrix.T @ x` by assembling the full matrix.
    """
    n = 12
    sigmas = [1., 2., 3.]
    k = len(sigmas)
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=sigmas)
    # Assemble forward matrix.
    mat = np.array([delta_norm.matvec(e_i) for e_i in np.eye(k * n)]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([delta_norm.rmatvec(e_i) for e_i in np.eye(k * n)]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()
    assert np.isclose(mat.T, adj_mat).all()


def test_spectrum():
    n = 100
    k = 10
    one_vec = np.ones(k)
    sigmas = 3 * np.arange(1, k+1)
    t_max = sigmas[-1]
    wth = 2.
    delta_norm = ScaleNormalizedLaplacian1D(size=n, sigmas=one_vec)
    nabla = ScaleNormalizedForwardDifference1D(size=n, sigmas=sigmas)
    a = nabla.csr_matrix @ delta_norm.csr_matrix
    ata = a.T @ a
    ata = ata.toarray()
    eig_max = np.linalg.norm(ata, ord=2)
    eig_min = np.linalg.norm(ata, ord=-2)
    print(f"Maximum eigenvalue: {eig_max}, minimal eigenvalue: {eig_min}")
    print(f"Eig_max / t^2: {eig_max / (t_max ** 2)}")
