import pytest
import numpy as np
from scipy.sparse import csr_matrix, spmatrix


from src.tvulog.differential_operators._scale_normalized_forward_difference1d import ScaleNormalizedForwardDifference1D
from src.tvulog.differential_operators._forward_difference import ForwardDifference
from src.tvulog.util import scale_discretization


RTOL = 1e-6     # Relative tolerance for numerical errors.
PLOT = False


def test_init():
    size = 123
    num_scales = 10
    sigmas = np.arange(1, 1 + num_scales, dtype=float)
    fdiff = ScaleNormalizedForwardDifference1D(size=size, sigmas=sigmas)
    assert fdiff.shape == (2 * num_scales * size, num_scales * size)


SIZE = 123
NUM_SCALES = 10


@pytest.fixture
def snfd1d():
    size = SIZE
    num_scales = NUM_SCALES
    sigmas = np.arange(1, 1 + num_scales, dtype=float)
    fdiff = ScaleNormalizedForwardDifference1D(size=size, sigmas=sigmas)
    return fdiff


def test_csr_matrix(snfd1d):
    size = SIZE
    num_scales = NUM_SCALES
    fdiff = snfd1d
    csr = fdiff.csr_matrix
    assert isinstance(csr, spmatrix)
    assert csr.shape == (2 * num_scales * size, num_scales * size)
    for csr_mat in fdiff.csr_matrix_list:
        assert isinstance(csr_mat, csr_matrix)


def test_fwd_arr(snfd1d):
    fdiff = snfd1d
    np.random.seed(42)
    test_arr = np.random.randn(NUM_SCALES, SIZE)
    fdiff_arr = fdiff.fwd_arr(test_arr)
    assert fdiff_arr.shape == (2, NUM_SCALES, SIZE)


def test_matrices():
    n = 10
    num_scales = 5
    sigma_min = 1.
    sigma_max = 10.
    sigmas = scale_discretization(sigma_min, sigma_max, num_scales)
    scales = np.square(sigmas)
    b = scales[1] / scales[0]
    scale_factor = 1 / (b - 1)
    nabla = ForwardDifference(shape=(num_scales, n), width_to_height=1.)
    nabla_norm = ScaleNormalizedForwardDifference1D(size=n, sigmas=sigmas)
    scale_diff, space_diff = nabla.csr_matrix_list
    scale_diff_norm, space_diff_norm = nabla_norm.csr_matrix_list
    scale_diff_mat = scale_diff.toarray()
    scale_diff_mat_norm = scale_diff_norm.toarray()
    difference1 = np.abs(scale_factor * scale_diff_mat - scale_diff_mat_norm)
    max_diff = np.max(difference1)
    assert max_diff <= 1e-2
    assert scale_diff_norm.shape == scale_diff.shape
    assert space_diff_norm.shape == space_diff.shape
    np.random.seed(42)
    v = np.random.randn(num_scales, n)
    g1 = (scale_diff_norm @ v.flatten()).reshape(num_scales, n)
    g2 = scale_factor * (scale_diff @ v.flatten()).reshape(num_scales, n)
    assert np.isclose(g1, g2).all()
    f1 = (space_diff_norm @ v.flatten()).reshape(num_scales, n)
    f2 = (space_diff @ v.flatten()).reshape(num_scales, n)
    f2 = f2 * sigmas[:, np.newaxis]
    assert np.isclose(f1, f2).all()
