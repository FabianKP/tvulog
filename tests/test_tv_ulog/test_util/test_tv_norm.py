
import numpy as np
import pytest

from src.tvulog.differential_operators import ForwardDifference
from src.tvulog.util import tv_norm, l1l2_norm


def test_one_dimension():
    """
    Tests TV-norm in 1D.
    """
    n = 100
    np.random.seed(42)
    arr = np.random.randn(n)
    nabla = ForwardDifference(shape=arr.shape, width_to_height=1.)
    nabla_arr = nabla @ arr
    tv_arr_ref = np.linalg.norm(nabla_arr, ord=1)
    tv_arr_test = tv_norm(arr)
    assert np.isclose(tv_arr_ref, tv_arr_test)


def test_two_dimensions():
    """
    Tests TV-norm in 2D.
    """
    m = 12
    n = 53
    np.random.seed(42)
    arr = np.random.randn(m, n)
    nabla = ForwardDifference(shape=arr.shape, width_to_height=1.)
    nabla_arr = nabla @ arr.flatten()
    tv_arr_ref = l1l2_norm(nabla_arr, d=2)
    tv_arr_test = tv_norm(arr)
    assert np.isclose(tv_arr_ref, tv_arr_test)


def test_three_dimensions():
    """
    Tests TV-norm in 3D.
    """
    k = 10
    m = 12
    n = 53
    np.random.seed(42)
    arr = np.random.randn(k, m, n)
    nabla = ForwardDifference(shape=arr.shape, width_to_height=1.)
    nabla_arr = nabla @ arr.flatten()
    tv_arr_ref = l1l2_norm(nabla_arr, d=3)
    tv_arr_test = tv_norm(arr)
    assert np.isclose(tv_arr_ref, tv_arr_test)


def test_four_dimensions():
    """
    For 4 dimensions, an error should be raised.
    """
    arr = np.random.randn(6, 5, 4, 3)
    with pytest.raises(NotImplementedError) as e_info:
        tv_norm(arr)

