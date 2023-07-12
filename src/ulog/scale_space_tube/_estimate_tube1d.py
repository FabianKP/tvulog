
import numpy as np
from typing import Optional, Sequence

from src.rectangular_cr import rectangular_cr

from ..gaussian_scale_space import scale_space_representation1d


def estimate_tube1d(alpha: float, samples: np.ndarray, sigmas: Sequence[float], neglogpdf: callable,
                        ref: Optional[np.ndarray] = None, verbose: Optional[bool] = False):
    """
    Estimates credible scale-space tube from Monte Carlo samples for a one-dimensional signal.

    Parameters
    ---
    alpha
        The credibility parameter.
    samples : shape (m, n)
        The Monte Carlo samples. `m` is the number of samples and `n` the signal size.
    sigmas :
        List of standard deviations for the Gaussian filter.
    width_to_height
        The width-to-height ratio for the Gaussian filter.
    neglogpdf :
        The negative logarithmic probability density.
    ref: shape (n,)
        A reference signal. The scale-space tube is computed in a way that ensures that it contains the
        scale-space representation of `ref`.
    verbose :
        If set to `True`, prints additional information during execution.

    Returns
    ---
    lower_bound : shape ( m, n)
        The lower bound of the scale-space tube.
    upper_stack : shape (m, n)
        The upper bound of the scale-space tube.
    """
    assert samples.ndim == 2
    m, n = samples.shape
    k = len(sigmas)
    # Compute log-pdfs.
    if verbose:
        print(f"Performing {samples.shape[0]} density evaluations.")
    g_vec = np.array([neglogpdf(x) for x in samples])
    def ssr_transform(x):
        return scale_space_representation1d(signal=x, sigmas=sigmas).flatten()

    # FIND RECTANGULAR CREDIBLE REGION.
    if verbose:
        print(f"Estimating credible scale-space tube.")
    lb, ub, theta_est = rectangular_cr(theta=1 - alpha, samples=samples, g=g_vec, mode=ref, transform=ssr_transform)
    # Reshape lb and ub into stack-form.
    lower_stack = np.reshape(lb, (k, n))
    upper_stack = np.reshape(ub, (k, n))

    return lower_stack, upper_stack