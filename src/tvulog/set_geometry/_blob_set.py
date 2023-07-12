
import numpy as np
from typing import Tuple


class BlobSet:
    """
    Container for the results of TV-ULoG. Represents a detected blob set, i.e. the projection of a mode set.
    """
    def __init__(self, shape: Tuple[int, ...], indices: Tuple[np.ndarray]):
        """
        Parameters
        ----------
        shape
            The shape of the domain where the blob set is detected.
        indices
            The indices defining the blob set.
        """
        assert len(indices) == len(shape)
        self.shape = shape
        self.indices = indices
