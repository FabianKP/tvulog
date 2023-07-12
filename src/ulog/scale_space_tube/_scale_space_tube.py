
import numpy as np


class ScaleSpaceTube:
    """
    Class that represents a scale space tube for a signal in vector form (1D).

    Members
    -------
    lower : np.ndarray, shape (k, n)
        The lower bound for the scale-space tube. Each row is a lower bound (vector) for the corresponding scale.
    upper : np.ndarray, shape (k, n)
        The upper bound for the scale-space tube. Each row is an upper bound (vector) for the corresponding scale.
    reference : np.ndarray, shape (k, n), optional
        The filtered value of a reference image. This should lie between `lower` and `upper`.
    num_scales : int
        The number of scales.
    num_pixels : int
        The size of the underlying signal.
    ref_inside : bool
        True if the reference image is inside the ScaleSpaceTube.
    """
    def __init__(self, lower: np.ndarray, upper: np.ndarray, reference: np.ndarray = None):
        """
        """
        assert upper.shape == lower.shape
        self.num_scales = lower.shape[0]
        self.num_pixels = lower.shape[1]
        self.reference = reference
        self.ref_inside = np.all(lower <= reference) and np.all(reference <= upper)
        if lower.shape[0] == 1:
            self.lower = lower.flatten()
            self.upper = upper.flatten()
        else:
            self.lower = lower
            self.upper = upper