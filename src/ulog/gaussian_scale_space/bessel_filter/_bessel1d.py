
import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d

from ._bessel_kernel import bessel_kernel


def bessel1d(image: np.ndarray, axis: int = 0, sigma: float = 1., truncate: float = 4.0):
    """
    Applies a one-dimensional Bessel filter along a given axis.
    Note that the filter computations are only stable for sigma < ~25.
    To stabilize the Bessel filter, we use the Gaussian filter for all sigma > 15.

    Parameters
    ---
    image : shape (m, n)
        The input image.
    axis : int
        The axis along which the filter is applied.
    sigma :
        The standard deviation of the filter.
    truncate :
        Truncate the kernel at this many standard deviations.
    """
    if sigma <= 15.:
        # Translate sigma to scale.
        t = sigma * sigma
        # Get truncated Bessel kernel.
        r_trunc = np.ceil(truncate * sigma).astype(int)
        g = bessel_kernel(t, r_trunc)
        # Convolve with image along given dimension.
        output = convolve1d(image, g, axis=axis, mode="mirror", cval=0.)
    else:
        output = gaussian_filter1d(image, sigma=sigma, axis=axis, mode="mirror", cval=0.)

    return output
