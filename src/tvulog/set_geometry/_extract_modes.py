
from dataclasses import dataclass
import numpy as np
import scipy as sp
from skimage import morphology

from ._mode_set import ModeSet


@dataclass
class LocalMaximum:
    """
    Object representing a local maximum of a discrete, univariate function.
    """
    ind: np.ndarray     # The index set representing the local maximizer.
    val: float          # The value of the local maximum.


def extract_modes(arr: np.ndarray, rthresh: float = 0.1, maxima_thresh: float = 0.01, eps_rel: float = 1e-4)\
        -> list[ModeSet]:
    """
    Given an N-dimensional input array, extracts mode sets. These are connected sets of indices corresponding to the
    local maxima/minima of the input array.

    In more detail, the method does the following:
    1. Detect the local maxima of the input array.
    2. Around each local maximizer, assembles the largest connected component of indices such that the
    corresponding array values are above a certain threshold.
    3. Returns each of these components as a `ModeSet` object.

    Parameters
    ----------
    arr
        The input array.
    rthresh : float
        The relative threshold. Must be a number between 0 and 1. An index is part of a mode set for a
        local maximum `y` if `input[y] >= rthresh * y`.
    maxima_thresh : float
        Relative threshold for what constitutes an extremum. An extremum is only detected if its intensity is at least
        `maxima_thresh` of the global maximum.
    eps_rel : float
        Small number used in comparisons.

    Returns
    -------
    mode_sets : list[ModeSet]
        The identified mode sets.
    """
    # Make input positive, i.e. `input.min() = 0.`.
    input_positive = arr - np.min(arr)
    eps = eps_rel * input_positive
    # Detect local maxima in input.
    local_maxima = _detect_local_maxima(arr=input_positive, rthresh=maxima_thresh)
    # Grow plateaus and store as `ModeSet` objects.
    mode_sets = []
    for local_max in local_maxima:
        plateau = _grow_plateau(arr=input_positive, local_max=local_max, rthresh=rthresh, eps=eps)
        mode = ModeSet(indices=plateau, value=local_max.val, rthresh=rthresh)
        if _is_proper_mode(mode=mode, arr=input_positive):
            mode_sets.append(mode)

    return mode_sets


def _detect_local_maxima(arr: np.ndarray, rthresh: float) -> list[LocalMaximum]:
    """
    Determines local maximizers of input array.

    Parameters
    ----------
    arr
        The input array. Must be non-negative!
    rthresh
        Relative threshold. A local maximum must be equal or larger to `rthresh * input.max()`, where `input.max()` is
        the global maximum.

    Returns
    -------
    local_maximizers_thresholded : list[LocalMaximum]
        The detected local maxima, represented as list of `LocalMaximum` instances.
    """
    # Determine local maxima.
    local_maximizers = morphology.local_maxima(image=arr, indices=True, allow_borders=True)
    local_maximizers = np.array(local_maximizers).T
    # Determine global maximum
    global_maximum = np.max(arr)
    # Remove local_maximizers below threshold and create `LocalMaximum` instances.
    local_maximizers_thresholded = []
    for maximizer in local_maximizers:
        val = arr[tuple(maximizer)]
        if val >= rthresh * global_maximum:
            local_maximum = LocalMaximum(ind=maximizer, val=val)
            local_maximizers_thresholded.append(local_maximum)

    return local_maximizers_thresholded


def _is_proper_mode(mode: ModeSet, arr: np.ndarray) -> bool:
    """
    Checks if a mode is a proper mode, i.e. if all neighbors are less than it.
    """
    # Determine set of neighbors.
    neighbors = _get_neighbors(ind=mode.indices, arr=arr)
    # Mode is proper only if no neighbor is larger than val.
    max_neighbor = np.max(arr[neighbors])
    mode_is_proper = (max_neighbor <= mode.value)
    return mode_is_proper


def _get_neighbors(ind: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    Returns set of neighbors of given index set.
    """
    mask = np.zeros_like(arr)
    mask[ind] = 1.
    neighbors = sp.ndimage.binary_dilation(input=mask)
    neighbors = neighbors - mask
    return np.where(neighbors > 0.)


def _grow_plateau(arr: np.ndarray, local_max: LocalMaximum, rthresh: float, eps: float) -> np.ndarray:
    """
    Given point-wise local maximum, detects the corresponding plateau, i.e. the largest connected component that
    contains the local maximum point, such that all values inside the component are close to the local maximum.

    Parameters
    ----------
    arr : shape (k, n) or (k, m, n)
        The input array.
    local_max
        The local maximum point.
    rthresh
        The relative threshold for a point to be inside the plateau (see the section "Visualization" in the paper).
    eps
        Small tolerance that is needed to account for numerical errors.

    Returns
    -------
    plateau
        The plateau given as an array of indices.
    """
    # Detect all connected components above threshold.
    labeled, nr_objects = sp.ndimage.label((arr >= rthresh * local_max.val) & (arr <= local_max.val + eps))
    # Identify the component that contains the local maximum.
    label_local_max = labeled[tuple(local_max.ind)]
    plateau = np.where(labeled == label_local_max)
    return plateau
