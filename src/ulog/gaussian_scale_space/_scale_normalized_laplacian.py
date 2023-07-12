
import numpy as np
from scipy.ndimage import generic_laplace, correlate1d
from typing import Sequence


def scale_normalized_laplacian(ssr: np.ndarray, sigmas: Sequence[float], width_to_height: float = 1) -> np.ndarray:
    """
    Computes the scale-normalized Laplacian of a given scale-space representation of an image.

    Parameters
    ----------
    ssr : shape (k, m, n)
        A scale-space object. The number `k` corresponds to the number of scales, while `m` and `n` are the image
        dimensions in vertical and horizontal direction.
    sigmas :
        The list of sigma-values. t := sigma ** 2.
    width_to_height :
        The width_to_height ratio , i.e. r = sigma_2 / sigma_1. Given `sigma`, the image is smoothed along the
        x2-direction (horizontal) by a Gaussian with standard deviation `sigma`, and along the x1-direction (vertically)
        by a Gaussian with standard deviation `sigma / r`.

    Returns
    -------
    snl : shape (k, m, n)
        The scale-normalized Laplacian of `ssr`.
    """
    # Check the input.
    assert ssr.ndim > 1
    assert ssr.shape[0] == len(sigmas)
    # Compute the scale-normalized Laplacian sequentially for all scales.
    snl_list = []
    for i in range(len(sigmas)):
        t_i = sigmas[i] ** 2
        snl_i = t_i * scaled_laplacian(image=ssr[i], width_to_height=width_to_height)
        snl_list.append(snl_i)
    snl = np.array(snl_list)
    # Check that the scale-normalized Laplacian has the same shape as the original scale-space rep.
    assert snl.shape == ssr.shape
    return snl


def scaled_laplacian(image: np.ndarray, width_to_height: float) -> np.ndarray:
    """
    Applies the scaled Laplacian operator. To realize the desired width-to-height ratio, it is applied to
    the image f_r(x1, x2) := f(T(x1, x2)) where T(x1, x2) = (x1 / r, x2). By the chain rule, we then have

    .. math::
        \\tilde{\\Delta} f(x1, x2) := \\Delta f_r(y1, y2)
                               = \\partial_{y_1}^2 f_r(y1, y2) + \\partial_{y_2}^2 f_r(y1, y2)
                               = \\partial_{x_1}^2 f(T^(-1)(y1, y2)) + \\partial_{x_2}^2 f(T^(-1)(y1, y2)
                               = r^2 \\partial_{x_1}^2 f(x1, x2) + \\partial_{x_2}^2 f(x1, x2).

    The implementation is based on scipy's generic_laplace filter.

    Parameters
    ---
    image : shape (m, n)
        The input image.
    width_to_height
        The width-to-height ratio `r`.

    Returns
    ---
    out : shape (m, n)
        The weighted Laplacian of `image`.
    """
    weights = np.array([width_to_height ** 2, 1.])

    def derivative_scaled(_image, _axis, _output, _mode, _cval):
        return weights[_axis] * correlate1d(_image, [1, -2, 1], _axis, _output, _mode, _cval, 0)
    return generic_laplace(image, derivative_scaled, mode="mirror")
