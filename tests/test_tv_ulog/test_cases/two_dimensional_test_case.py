
from matplotlib import pyplot as plt
import numpy as np
import os
from skimage.filters import gaussian

from src.tvulog.differential_operators import forward_difference



dirname = os.path.dirname(__file__)


def two_dimensional_test_case():
    """
    Two-dimensional test problem for constrained TV minimization.

    Returns
    ---
    lower2d : shape (m, n)
        The pixel-wise lower bounds.
    upper2d : shape (m, n)
        The pixel-wise upper bounds.
    map2d : shape (m, n)
        The pixel-wise MAP estimate.
    """
    lb_path = os.path.join(dirname, 'data/lb.npy')
    ub_path = os.path.join(dirname, 'data/ub.npy')
    map_path = os.path.join(dirname, 'data/filtered_map.npy')
    lower2d = np.load(str(lb_path))
    upper2d = np.load(str(ub_path))
    map2d = np.load(str(map_path))
    return lower2d, upper2d, map2d


def gaussian_test(n: int):
    # Create basic image.
    np.random.seed(42)
    sigma = n / 10
    img = np.zeros((n, n))
    # Set center pixel equal to 1.
    center = (int(n / 2), int(n / 2))
    img[center] = 1.
    # Filter to get Gaussian.
    img = gaussian(img, sigma=sigma)
    # Rescale
    img = img / np.max(img)
    sigma = .2
    delta = 0.1
    ub = img + delta + np.abs(sigma * np.random.randn(n, n))
    lb = img - delta - np.abs(sigma * np.random.randn(n, n))
    # Apply slight smoothing.
    epsilon = n / 100
    lb = gaussian(lb, epsilon)
    ub = gaussian(ub, epsilon)
    return lb, img, ub


def gaussian_deriv_test(axis: int, n: int):
    # Create basic image.
    np.random.seed(42)
    sigma = n / 10
    img = np.zeros((n, n))
    # Set center pixel equal to 1.
    center = (int(n / 2), int(n / 2))
    img[center] = 1.
    # Filter to get Gaussian.
    img = gaussian(img, sigma=sigma)
    img = (forward_difference(img, axis)).reshape(n, n)
    # Rescale
    vmax = np.max(np.abs(img))
    img = img / vmax
    sigma = .2
    delta = 0.2
    ub = img + delta + np.abs(sigma * np.random.randn(n, n))
    lb = img - delta - np.abs(sigma * np.random.randn(n, n))
    # Apply slight smoothing.
    epsilon = n / 100
    lb = gaussian(lb, epsilon)
    ub = gaussian(ub, epsilon)
    return lb, img, ub

def test_gaussian():
    n = 50
    lb, img, ub = gaussian_test(n)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(lb)
    ax[1].imshow(img)
    ax[2].imshow(ub)
    fig2, ax2 = plt.subplots(1, 1)
    i = int(n / 2)
    ax2.plot(lb[i])
    ax2.plot(ub[i])
    ax2.plot(img[i])
    plt.show()


def test_gaussian_deriv():
    n = 50
    lb, img, ub = gaussian_deriv_test(1, n)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(lb)
    ax[1].imshow(img)
    ax[2].imshow(ub)
    fig2, ax2 = plt.subplots(1, 1)
    i = int(n / 2)
    ax2.plot(lb[:, i])
    ax2.plot(ub[:, i])
    ax2.plot(img[:, i])
    plt.show()


def test_two_dimensional_test_case():
    horizontal_slice_index = 5
    lb, ub, map_estimate = two_dimensional_test_case()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(lb)
    ax[1].imshow(map_estimate)
    ax[2].imshow(ub)
    ax[3].plot(lb[horizontal_slice_index])
    ax[3].plot(map_estimate[horizontal_slice_index])
    ax[3].plot(ub[horizontal_slice_index])
    plt.show()