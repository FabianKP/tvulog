
import numpy as np
from typing import Sequence

from ._scale_normalized_forward_difference1d import ScaleNormalizedForwardDifference1D
from ._scale_normalized_forward_difference2d import ScaleNormalizedForwardDifference2D
from ..util import l1l2_norm


def scale_normalized_total_variation(arr: np.ndarray, sigmas: Sequence[float], width_to_height: float) -> float:
    """
    Computes the scale-normalized total variation of a 2- or 3-dimensional scale-space object.
    That is, it computes

    .. math::
        \\text{TV}(X) = \\sum_i ||(\\nabla X)_i||,

    for an array `X`.

    Parameters
    ---
    arr : shape (k, n) or (k, m, n)
        Two-dimensional (corresponding to 1D-signals) or three-dimensional (corresponding to images) scale-space object.
    sigmas
        Standard deviations for the scale-space representation.
    width_to_height
        Weighting between horizontal and vertical axis.

    Returns
    ---
    tv_norm
        The value of the scale-normalized total variation of `arr`.
    """
    if arr.ndim == 2:
        # Create scale-normalized forward difference.
        nabla_norm = ScaleNormalizedForwardDifference1D(size=arr.shape[1], sigmas=sigmas)
        nabla_norm_arr = nabla_norm.fwd_arr(arr)    # outputs (2, k, n)-array.
        # Reshape so that we can apply l1-norm.
        nabla_norm_arr = nabla_norm_arr.reshape((2, -1,), order="F")
        tv_norm = l1l2_norm(nabla_norm_arr, d=2)
    elif arr.ndim == 3:
        # Create scale-normalized forward difference.
        nabla_norm = ScaleNormalizedForwardDifference2D(m=arr.shape[1], n=arr.shape[2], sigmas=sigmas,
                                                        width_to_height=width_to_height)
        nabla_norm_arr = nabla_norm.fwd_arr(arr)  # outputs (3, k, m, n)-array.
        # Reshape so that we can apply l1-norm.
        nabla_norm_arr = nabla_norm_arr.reshape((3, -1,), order="F")
        tv_norm = l1l2_norm(nabla_norm_arr, d=3)
    else:
        raise ValueError("Only works for 2- and 3-dimensional scale-space objects.")
    return tv_norm
