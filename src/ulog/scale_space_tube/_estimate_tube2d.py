
import numpy as np
from typing import Optional, Sequence

from src.rectangular_cr import rectangular_cr

from ..gaussian_scale_space import scale_space_representation2d


def estimate_tube2d(alpha: float, samples: np.ndarray, sigmas: Sequence[float], width_to_height: float,
                        neglogpdf: callable, ref_image: Optional[np.ndarray] = None, verbose: Optional[bool] = None):
    """
    Computes stack of FCIs for different scales.

    Parameters
    ---
    alpha
        The credibility parameter.
    samples : shape (k, m, n)
        The Monte Carlo samples. `k` is the number of samples and (m, n) their shape.
    sigmas :
        List of standard deviations for the Gaussian filter.
    width_to_height
        The width-to-height ratio for the Gaussian filter.
    neglogpdf : shape (m, n)
        The negative logarithmic probability density.
    ref_image : shape (m, n)
        A reference image. The scale-space tube is computed in a way that ensures that it contains the
        scale-space representation of `ref`.
    verbose : bool
        If set to `True`, prints additional information during execution.

    Returns
    ---
    lower_bound : shape (s, m, n)
        The lower bound of the scale-space tube. Here, `s = len(sigmas)` is the number of scales.
    upper_bound : shape (s, m, n)
        The upper bound of the scale-space tube. Here, `s = len(sigmas)` is the number of scales.
    """
    assert samples.ndim == 3
    # Compute log-pdfs.
    if verbose:
        print(f"Performing {samples.shape[0]} density evaluations.")
    g_vec = np.array([neglogpdf(x) for x in samples])
    s, m, n = samples.shape
    flattened_samples = samples.reshape(s, m * n)
    k = len(sigmas)

    def ssr_transform(x):
        x_im = np.reshape(x, (m, n))
        ssr = scale_space_representation2d(image=x_im, sigmas=sigmas, width_to_height=width_to_height)
        return ssr.flatten()

    # ESTIMATE CREDIBLE SCALE SPACE TUBE.
    if verbose:
        print(f"Estimating credible scale-space tube.")
    lb, ub, theta_est = rectangular_cr(theta=1 - alpha, samples=flattened_samples, g=g_vec, mode=ref_image.flatten(),
                                       transform=ssr_transform)

    # Reshape lower and upper bound to scale-space tube format.
    lower_bound = np.array(lb.reshape(k, m, n))
    upper_bound = np.array(ub.reshape(k, m, n))

    return lower_bound, upper_bound
