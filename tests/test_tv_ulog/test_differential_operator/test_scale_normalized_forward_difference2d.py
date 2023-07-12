
import pytest
import numpy as np
from scipy.sparse import spmatrix, csr_matrix

from src.tvulog.differential_operators._scale_normalized_forward_difference2d import ScaleNormalizedForwardDifference2D
from src.tvulog.differential_operators._forward_difference import ForwardDifference
from src.tvulog.util import scale_discretization


RTOL = 1e-6     # Relative tolerance for numerical errors.
PLOT = False

M = 12
N = 53
NUM_SCALES = 10
WTH = 2.


def test_init():
    sigmas = np.arange(1, 1 + NUM_SCALES, dtype=float)
    fdiff = ScaleNormalizedForwardDifference2D(m=M, n=N, sigmas=sigmas, width_to_height=WTH)
    assert fdiff.shape == (3 * NUM_SCALES * M * N, NUM_SCALES * M * N)
    assert fdiff.x_shape == (NUM_SCALES, M, N)
    assert fdiff.y_shape == (3, NUM_SCALES, M, N)


@pytest.fixture
def snfd2d():
    sigmas = np.arange(1, 1 + NUM_SCALES, dtype=float)
    fdiff = ScaleNormalizedForwardDifference2D(m=M, n=N, sigmas=sigmas, width_to_height=WTH)
    return fdiff


def test_csr_matrix(snfd2d):
    fdiff = snfd2d
    csr = fdiff.csr_matrix
    assert isinstance(csr, spmatrix)
    assert csr.shape == (3 * NUM_SCALES * M * N, NUM_SCALES * M * N)
    for csr_mat in fdiff.csr_matrix_list:
        assert isinstance(csr_mat, csr_matrix)


def test_fwd_arr(snfd2d):
    fdiff = snfd2d
    np.random.seed(42)
    test_arr = np.random.randn(NUM_SCALES, M, N)
    fdiff_arr = fdiff.fwd_arr(test_arr)
    assert fdiff_arr.shape == (3, NUM_SCALES, M, N)


def test_scale_normalization():
    sigma_min = 1.
    sigma_max = 10.
    sigmas = scale_discretization(sigma_min=sigma_min, sigma_max=sigma_max, num_scales=NUM_SCALES)
    scales = np.square(sigmas)
    b = scales[1] / scales[0]
    scale_factor = 1 / (b - 1)
    normalized_fdiff = ScaleNormalizedForwardDifference2D(m=M, n=N, sigmas=sigmas, width_to_height=WTH)
    unnormalized_fdiff = ForwardDifference(shape=(NUM_SCALES, M, N), width_to_height=WTH)
    np.random.seed(42)
    test_arr = np.random.randn(NUM_SCALES, M, N)
    test_vec = test_arr.flatten()
    x_normalized = (normalized_fdiff @ test_vec).reshape((3, NUM_SCALES, M, N))
    x_unnormalized = (unnormalized_fdiff @ test_vec).reshape((3, NUM_SCALES, M, N))
    # scale derivatives should be the same.
    f1 = x_normalized[0]
    f2 = x_unnormalized[0]
    f2 = scale_factor * f2
    assert np.isclose(f1, f2).all()
    # space derivatives should be the same after normalization
    g1 = x_normalized[1]
    g2 = x_unnormalized[1]
    g2 = g2 * sigmas[:, np.newaxis, np.newaxis]
    assert np.isclose(g1, g2).all()
    h1 = x_normalized[2]
    h2 = x_unnormalized[2]
    h2 = h2 * sigmas[:, np.newaxis, np.newaxis]
    assert np.isclose(h1, h2).all()