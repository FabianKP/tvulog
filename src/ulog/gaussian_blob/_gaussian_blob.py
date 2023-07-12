
from math import sqrt
import numpy as np


class GaussianBlob:
    """
    Represents a two-dimensional Gaussian blob.
    """
    def __init__(self, x1: int, x2: int, sigma: float, width_to_height: float, log: float):
        """
        We always use the axes
        0----> x2
        |
        |
        v
        x1

        Parameters
        ----------
        x1 : int
            The vertical position of the blob center.
        x2 : int
            The horizontal position of the blob center.
        sigma :
            The horizontal standard deviation associated to the blob.
            The vertical standard deviation is `sigma / width_to_height`.
        width_to_height :
            The width-to-height ratio of the blob. A large value corresponds to flat blobs, a small value to tall blobs.
        """
        self._x1 = x1
        self._x2 = x2
        self._sigma = sigma
        self._sigma1 = sigma / width_to_height
        self._sigma2 = sigma
        self._log = log

    @property
    def position(self) -> np.ndarray:
        """
        Returns the position [x, y] of the blob.
        """
        return np.array([self._x1, self._x2])

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def x2(self) -> int:
        return self._x2

    @property
    def width(self) -> float:
        """
        The horizontal width of the blob. Since the horizontal radius r_y satisfies :math:`r_y = \\sqrt{2}\\sigma_y',
        the width, which is two-times the radius, is given by :math:'w = 2 \\sqrt{2} \\sigma_y`.
        """
        return 2 * sqrt(2) * self._sigma2

    @property
    def height(self) -> float:
        """
        The vertical height of the blob. It is given by 2 * sqrt(2) * sigma_x.
        """
        return 2 * sqrt(2) * self._sigma1

    @property
    def vector(self) -> np.ndarray:
        """
        The vector representation [x1, x2, sigma1, sigma2] of the blob.
        """
        return np.array([self._x1, self._x2, self._sigma1, self._sigma2])

    @property
    def log(self) -> float:
        """
        The value of the blob's scale-space Laplacian.
        """
        return self._log

    @property
    def scale(self) -> float:
        """
        The scale of the blob is simply given by sigma ** 2.
        """
        return self._sigma ** 2
