
import numpy as np
from typing import Sequence

from ..gaussian_scale_space import bessel1d, bessel2d


def scale_space_representation1d(signal: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """
    Computes the discrete Gaussian scale-space representation of a given one-dimensional signal.

    Parameters
    ---
    signal : shape (n, )
    sigmas :
        The standard deviations.

    Returns
    ---
    ssr : shape (len(sigmas), n)
        The discrete scale-space representation of `signal`.
    """
    ssr = np.array([bessel1d(image=signal, axis=0, sigma=s) for s in sigmas])
    return ssr


def scale_space_representation2d(image: np.ndarray, sigmas: Sequence[float], width_to_height: float = 1.) -> np.array:
    """
    Computes the discrete Gaussian scale-space representation of a given image.

    Parameters
    ----------
    image : shape (m, n)
        The input image.
    sigmas :
        The list of standard deviations for which the scale-space rep. is computed.
    width_to_height :
        The width_to_height ratio , i.e. r = sigma_2 / sigma_1. Given `sigma`, the image is smoothed along the
        x2-direction (horizontal) by a Gaussian with standard deviation `sigma`, and along the x1-direction (vertically)
        by a Gaussian with standard deviation `sigma / r`.

    Returns
    -------
    ssr : shape (k, m, n)
        The discrete scale-space representation of `image`.
    """
    width_to_height = float(width_to_height)
    assert isinstance(width_to_height, float) and width_to_height > 0
    scaled_images = []
    for sigma in sigmas:
        sigma_vec = np.array([sigma / width_to_height, sigma])
        scaled_image = bessel2d(image, sigma=sigma_vec)
        scaled_images.append(scaled_image)
    ssr = np.array(scaled_images)
    return ssr
