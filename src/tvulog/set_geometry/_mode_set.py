
import numpy as np


class ModeSet:
    """
    Represents a set in scale space where a function attains a mode, i.e. a connected component of indices.
    Note that the connectedness is not explicitly checked but implicitly assumed.
    """
    def __init__(self, indices: np.ndarray, value: float, rthresh: float):
        """
        Parameters
        ----------
        indices : shape (d, n)
            The indices defining the mode.
        value
            The value of the mode.
        rthresh
            The used relative threshold.
        """
        # Check dimension.
        index_arr = np.array(indices)
        assert index_arr.ndim == 2
        self.dim = index_arr.shape[0]
        self.indices = indices
        self.value = value
        self.rthresh = rthresh
