
import numpy as np

from ..differential_operators import ForwardDifference
from ._l1l2_norm import l1l2_norm


def tv_norm(x: np.ndarray, width_to_height: float = 1.) -> float:
    """
    Computes the discrete isotropic total variation norm of a one-, two-, or three-dimensional array.

    Parameters
    ----------
    x
        The input array. Must satisfy `1 <= x.ndim <= 3`.

    Returns
    -------
    tv
        The isotropic total variation of `x`.
    """
    if not 1 <= x.ndim <= 3:
        raise NotImplementedError("'x' must be a 1-, 2-, or 3-dimensional numpy array.")
    ndim = x.ndim
    nabla = ForwardDifference(shape=x.shape, width_to_height=width_to_height)
    x_flat = x.flatten()
    nabla_x_flat = nabla.matvec(x_flat)
    tv_x = l1l2_norm(nabla_x_flat, ndim)
    return tv_x




