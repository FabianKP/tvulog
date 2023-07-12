
import numpy as np

from src.tvulog.differential_operators._scale_normalized_laplacian2d import ScaleNormalizedLaplacian2D
from src.tvulog.differential_operators._laplacian2d import Laplacian2D
from src.tvulog.differential_operators._scale_normalized_forward_difference2d import ScaleNormalizedForwardDifference2D


np.random.seed(42)


def test_construct():
    m = 12
    n = 53
    sigmas = [1, 2, 3]
    k = len(sigmas)
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=1.)
    delta_norm_mat = delta_norm.csr_matrix
    assert delta_norm.shape == (k * m * n, k * m * n)
    assert delta_norm.x_size == k * m * n
    assert delta_norm_mat.shape == (k * m * n, k * m * n)


def test_print_csr_mat():
    m = 4
    n = 4
    sigmas = [1, 2, 3]
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=1)
    csr_mat = delta_norm.csr_matrix
    print("\n")
    print(csr_mat.toarray())


def test_fwd_action():
    """
    Tests on random vectors if the forward action of the scale-normalized Laplacian is correct.
    """
    np.random.seed(42)  # makes test deterministic
    num_tests = 10
    width_to_height = 0.5
    m = 12
    n = 53
    k = 10
    sigmas = np.arange(1, 1 + k)
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
    delta = Laplacian2D(m=m, n=n, width_to_height=width_to_height)
    scales = np.square(sigmas)
    for _ in range(num_tests):
        x = np.random.randn(k, m * n)
        # Compute scale-normalized Laplacian manually
        delta_norm_x_ref = [t * delta.matvec(x_i) for t, x_i in zip(scales, x)]
        delta_norm_x_ref = np.array(delta_norm_x_ref)
        # Now, compute with `matvec`.
        delta_norm_x = delta_norm.matvec(x.flatten())
        delta_norm_x = delta_norm_x.reshape(k, m * n)
        assert np.isclose(delta_norm_x, delta_norm_x_ref).all()


def test_adj_action():
    """
    Tests on random vectors if the adjoint action of the scale-normalized Laplacian is correct.
    Note that the adjoint of the scale-normalized Laplacian is simply the scale-normalized adjoint Laplacian!
    """
    np.random.seed(42)  # makes test deterministic
    num_tests = 10
    width_to_height = 0.5
    m = 12
    n = 53
    k = 10
    sigmas = np.arange(1, 1 + k)
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
    delta = Laplacian2D(m=m, n=n, width_to_height=width_to_height)
    scales = np.square(sigmas)
    for _ in range(num_tests):
        x = np.random.randn(k, m * n)
        # Compute adjoint scale-normalized Laplacian manually
        adj_delta_norm_x_ref = [t * delta.rmatvec(x_i) for t, x_i in zip(scales, x)]
        adj_delta_norm_x_ref = np.array(adj_delta_norm_x_ref)
        # Now, compute with `matvec`.
        adj_delta_norm_x = delta_norm.rmatvec(x.flatten())
        adj_delta_norm_x = adj_delta_norm_x.reshape(k, m * n)
        assert np.isclose(adj_delta_norm_x, adj_delta_norm_x_ref).all()


def test_adjoint_matrix():
    """
    Check that `self.rmatvec(x)` truly corresponds to `self.csr_matrix.T @ x` by assembling the full matrix.
    """
    m = 12
    n = 10
    sigmas = [1., 2., 3.]
    width_to_height = 1.
    k = len(sigmas)
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
    # Assemble forward matrix.
    mat = np.array([delta_norm.matvec(e_i) for e_i in np.eye(k * m * n)]).T
    # Assemble adjoint matrix.
    adj_mat = np.array([delta_norm.rmatvec(e_i) for e_i in np.eye(k * m * n)]).T
    # Compare
    assert np.isclose(mat.T, adj_mat).all()
    assert np.isclose(mat.T, adj_mat).all()


def test_spectrum():
    m = 20
    n = 10
    k = 5
    one_vec = np.ones(k)
    sigmas = 3 * np.arange(1, k+1)
    t_max = sigmas[-1]
    wth = 2.
    delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=one_vec, width_to_height=wth)
    nabla = ScaleNormalizedForwardDifference2D(m=m, n=n, sigmas=sigmas, width_to_height=wth)
    a = nabla.csr_matrix @ delta_norm.csr_matrix
    ata = a.T @ a
    ata = ata.toarray()
    eig_max = np.linalg.norm(ata, ord=2)
    eig_min = np.linalg.norm(ata, ord=-2)
    print(f"Maximum eigenvalue: {eig_max}, minimal eigenvalue: {eig_min}")
    print(f"Eig_max / t^2: {eig_max / (t_max ** 2)}")

