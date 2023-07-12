
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import gaussian


NUM_SCALES = 10


def tv_ulog1d_test_case(n: int):
    """
    Test case for TV-ULoG optimization problem for one-dimensional signals, i.e. for a problem of the form

    min_x ||D L x||_1 s. t. lb <= x <= ub,

    where `D` is the forward-difference operator and `L` is the scale-normalized Laplacian for 1D signals.
    The test case is faked by first generating a lowest scale test case, and then filtering both the lower bound,
    upper bound and reference with the appropriate number of scales.

    Parameters
    ----------
    n
        The desired signal dimension.

    Returns
    -------
    lb : shape (K, N)
        The lower bound FCIs.
    ub : shape (K, N)
        The upper bound FCIs.
    reference : shape (N, )
        The reference solution.
    sigmas : list[float]
        The used sigma values.
    """

    sigmas = 0.5 + np.arange(NUM_SCALES)
    # Create the one-dimensional basis problem.
    lb, ub, ref = _basis1d(n)
    # Filter everything to create fake FCIs. Also simulate convergence
    lower_fcis = np.array([gaussian(lb, sigma=s) + 0.05 * s for s in sigmas])
    upper_fcis = np.array([gaussian(ub, sigma=s) - 0.05 * s for s in sigmas])
    return lower_fcis, upper_fcis, ref, sigmas


def _basis1d(n: int):
    """
    Creates one-dimensional lower bound, upper bound and reference.
    """
    x_range = np.linspace(-5, 5, n)
    n = x_range.size
    sine = np.sin(x_range)
    eta = .5
    delta = 0.8
    ub = sine + delta + np.abs(eta * np.random.randn(n))
    lb = sine - delta - np.abs(eta * np.random.randn(n))
    # Apply slight smoothing.
    epsilon = n / 500
    lb = gaussian(lb, epsilon)
    ub = gaussian(ub, epsilon)
    # Reference is simply the average of `ub` and `lb`.
    ref = 0.5 * (lb + ub)
    return lb, ub, ref


def test_tv_ulog1d_test_case():
    lb, ub, ref, sigmas = tv_ulog1d_test_case(n=300)
    k = len(sigmas)
    # Visualize filtered credible intervals.
    fig, ax = plt.subplots(1, k, figsize=(15, 3))
    filtered_reference = np.array([gaussian(ref, sigma=s) for s in sigmas])
    for i in range(k):
        ax[i].plot(filtered_reference[i])
        ax[i].plot(lb[i])
        ax[i].plot(ub[i])
        ax[i].set_title(f"sigma = {sigmas[i]}")
    plt.show()