
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import gaussian


def noisy_sine(n):
    """
    One-dimensional test problem for constrained TV minimization.
    """
    np.random.seed(42)
    x_range = np.linspace(-5, 5, n)
    n = x_range.size
    sine = np.sin(x_range)
    sigma = .5
    delta = 0.3
    ub = sine + delta + np.abs(sigma * np.random.randn(n))
    lb = sine - delta - np.abs(sigma * np.random.randn(n))
    # Apply slight smoothing.
    epsilon = n / 200
    lb = gaussian(lb, epsilon)
    ub = gaussian(ub, epsilon)
    return lb, sine, ub


def test_noisy_sine():
    n = 300
    lb, ref, ub = noisy_sine(n=n)
    x_range = np.linspace(-5, 5, n)
    plt.plot(x_range, ref)
    plt.plot(x_range, ub)
    plt.plot(x_range, lb)
    plt.show()