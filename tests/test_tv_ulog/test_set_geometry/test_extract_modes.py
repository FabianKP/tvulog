
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tvulog.set_geometry import extract_modes, ModeSet
from src.tvulog.set_geometry._extract_modes import _detect_local_maxima, LocalMaximum, _grow_plateau, _get_neighbors
from src.tvulog.differential_operators import ForwardDifference
from src.tvulog.optimization._primal_smoothing import primal_smoothing_fpg
from tests.test_tv_ulog.test_cases import noisy_sine, two_dimensional_test_case


PLOT = False


@pytest.fixture()
def taut_string_example():
    """
    Test example for mode-extraction based on taut string.
    """
    n = 1000
    # Get sine example.
    lb, truth, ub = noisy_sine(n)
    # Determine taut-string solution.
    nabla = ForwardDifference(shape=(n,), width_to_height=1.)
    a_op = nabla
    max_eig = 4.
    mu = 1.
    sol = primal_smoothing_fpg(a=a_op, mu=mu, lb=lb, ub=ub, x0=truth, stepsize=1 / max_eig, max_iter=10000)
    taut_string = sol.x
    taut_string_deriv = nabla @ taut_string
    return taut_string_deriv

@pytest.fixture()
def two_dim_example():
    """
    Test example for mode-extraction based on taut string.
    """
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant2d.npy")
    return x

@pytest.fixture()
def three_dim_example():
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant3d.npy")
    x = - x
    x = x.clip(min=0.)
    return x


def test_taut_string_example(taut_string_example):
    x = taut_string_example
    if PLOT:
        plt.plot(x)
        plt.show()


def test_two_dim_example(two_dim_example):
    x = two_dim_example
    if PLOT:
        plt.imshow(x)
        plt.show()


def test_three_dim_example(three_dim_example):
    x_stack = three_dim_example
    k = x_stack.shape[0]
    if PLOT:
        fig, ax = plt.subplots(k, 1, figsize=(6, 18))
        vmax = np.max(x_stack)
        vmin = np.min(x_stack)
        for i in range(k):
            ax[i].imshow(x_stack[i], cmap="gnuplot", vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.show()


def test_detect_local_maxima(taut_string_example):
    x = taut_string_example
    x = x - np.min(x)
    maxima = _detect_local_maxima(arr=x, rthresh=0.01)
    for maximum in maxima:
        assert isinstance(maximum, LocalMaximum)
    if PLOT:
        plt.plot(x, color="b", label="input")
        max_inds = [max.ind for max in maxima]
        max_vals = [max.val for max in maxima]
        plt.scatter(max_inds, max_vals, color="r", label="local maximum")
        plt.legend()
        plt.show()


def test_grow_plateau(taut_string_example):
    x = taut_string_example
    x = x - np.min(x)
    # Rescale
    x = x / np.max(x)
    # Take the local maximum for the peak in the middle.
    ind = np.array([500])
    local_max = LocalMaximum(ind=ind, val=x[ind])
    plateau = _grow_plateau(arr=x, local_max=local_max, rthresh=0.1, eps=1e-3)
    if PLOT:
        plt.plot(x, color="b", label="input")
        # Visualize plateau with indicator
        indicator = np.zeros_like(x)
        indicator[plateau] = 1.
        plt.plot(indicator, color="r", linestyle=":", label="plateu")
        plt.legend()
        plt.show()


def test_get_neighbors():
    # Make test image.
    lb, ub, x = two_dimensional_test_case()
    x_max = np.max(x)
    x_thresh = (x >= 0.5 * x_max)
    ind = np.where(x_thresh > 0.)
    # Get neighbors
    neighbors = _get_neighbors(ind=ind, arr=x)
    # Visualize neighbors
    neighbor_img = np.zeros_like(x)
    neighbor_img[neighbors] = 1.
    if PLOT:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(x_thresh)
        ax[1].imshow(neighbor_img)
        plt.show()


def test_extract_modes_one_dimensional(taut_string_example):
    x = taut_string_example
    # Rescale to 0-1 scale
    x = x - np.min(x)
    x = x / np.max(x)
    modes = extract_modes(arr=x, rthresh=0.5)
    # Check that modes are ModeSets.
    for mode in modes:
        assert isinstance(mode, ModeSet)
    # Visualize modes.
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, color="b", label="input")
        # Visualize modes.
        for mode in modes:
            mode_height = np.ones_like(mode.indices) * mode.value
            ax.scatter(mode.indices, mode_height, color="r")
        plt.legend()
        plt.show()


def test_extract_modes_two_dimensional(two_dim_example):
    x = two_dim_example
    modes = extract_modes(arr=x, rthresh=0.5)
    # Check that modes are ModeSets.
    for mode in modes:
        assert isinstance(mode, ModeSet)
    # Visualize modes.
    if PLOT:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(x)
        # Visualize modes.
        mode_img = np.zeros_like(x)
        for mode in modes:
            mode_img[mode.indices] = mode.value
        ax[1].imshow(mode_img)
        plt.show()


def test_extract_modes_three_dimensional(three_dim_example):
    x_stack = three_dim_example
    k = x_stack.shape[0]
    modes = extract_modes(arr=x_stack, rthresh=0.8)
    # Check that modes are ModeSets.
    for mode in modes:
        assert isinstance(mode, ModeSet)
    # Make indicator function for mode stack.
    mode_stack = np.zeros_like(x_stack)
    for mode in modes:
        mode_stack[mode.indices] = mode.value
    mmin = np.min(mode_stack)
    mmax = np.max(mode_stack)
    if PLOT:
        fig, ax = plt.subplots(k, 2, figsize=(12, 20))
        vmax = np.max(x_stack)
        vmin = np.min(x_stack)
        for i in range(k):
            ax[i, 0].imshow(x_stack[i], cmap="gnuplot", vmin=vmin, vmax=vmax)
            ax[i, 1].imshow(mode_stack[i], cmap="gnuplot", vmin=mmin, vmax=mmax)
        plt.tight_layout()
        plt.show()





