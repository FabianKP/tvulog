
import numpy as np


def scale_discretization(sigma_min: float, sigma_max: float, num_scales: int) -> np.ndarray:
    """
    Computes an exponential scale discretization. That is, returns exponentially increasing sigma-values, i.e.
    `sigma[k+1] = b * sigma[k]`, such that `sigma[0] = sigma_min` and `sigma[num_scales - 1] = sigma_max`. The last
    equality might not hold exactly due to numerical errors.

    Parameters
    ----------
    sigma_min
        Lower bound for sigma interval.
    sigma_max
        Upper bound for sigma interval.
    num_scales
        Number of desired scales. Must be at least 2.

    Returns
    -------
    Scale discretization as numpy array.
    """
    if num_scales < 2:
        raise ValueError("`num_scales` must be at least 2.")
    k = num_scales - 1
    # Compute b such that `sigma_max = b^k sigma_min`.
    b = np.power(sigma_max / sigma_min, 1 / k)
    sigmas = np.array([np.power(b, i) * sigma_min for i in range(num_scales)])
    return sigmas