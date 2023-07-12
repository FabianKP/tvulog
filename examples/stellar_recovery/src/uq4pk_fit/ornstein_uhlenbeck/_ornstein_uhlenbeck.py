
from copy import deepcopy
import numpy as np
import scipy.linalg as sl
from typing import Union

from ..regop import RegularizationOperator


class OrnsteinUhlenbeck(RegularizationOperator):
    """
    Implementation of the Ornstein-Uhlenbeck covariance operator as `RegularizationOperator`.
    It expresses the intuitive assumption that pixels in an image are positively correlated, but that this
    correlation decreases exponentially in scale. See also our paper
    "Uncertainty-Aware Blob Detection with an Application to Integrated-Light Stellar Population Recoveries".
    """
    def __init__(self, m, n, h: Union[float, np.ndarray]):
        """

        Parameters
        ----------
        m
            Number of image rows.
        n
            Number of image columns.
        h
            Correlation length. If `h` is a float, the same correlation length is used in vertical and horizontal
            direction. If `h` is a 2-vector, the first entry gives the correlation length in vertical, and the
            second entry the correlation length in horizontal direction.
        """
        self._m = m
        self._n = n
        self._h = deepcopy(h)     # h can be either float or 2-vector!
        self._dim = m * n
        self._rdim = self.dim
        self.cov = self._compute_cov()
        self._imat = np.array(sl.sqrtm(self.cov))
        mat = np.linalg.inv(self._imat)
        RegularizationOperator.__init__(self, mat)

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    def fwd(self, v: np.ndarray) -> np.ndarray:
        return self._mat @ v

    def adj(self, v: np.ndarray) -> np.ndarray:
        return self.mat.T @ v

    def inv(self, w: np.ndarray) -> np.ndarray:
        return self._imat @ w

    def _compute_cov(self) -> np.ndarray:
        """
        Computes the Ornstein-Uhlenbeck covariance matrix with given size and correlation length.
        The entries of the matrix are given by
        cov[i,j] = exp(-||pos(i) - pos(j)||/h),
        where pos(i) is the normalized position of the i-th pixel.

        Returns
        -------
        cov
            The Ornstein-Uhlenbeck covariance matrix, a numpy array of shape (mn, mn).
        """
        # This function is simply a vectorized implementation of the above index-wise formula.
        # First, we compute the vector of normalized positions for all dim*dim-1 pixels.
        p = np.zeros((self._m * self._n, 2))
        for i in range(self._m * self._n):
            p[i, :] = self._pos(i) / self._h
        pdiff0 = np.subtract.outer(p[:, 0], p[:, 0])
        pdiff1 = np.subtract.outer(p[:, 1], p[:, 1])
        pdiff = np.dstack((pdiff0, pdiff1))
        diffnorm = np.linalg.norm(pdiff, axis=2)
        cov = np.exp(-diffnorm)
        assert cov.shape == (self._dim, self._dim)
        return cov

    def _pos(self, i) -> np.ndarray:
        """
        Given a picture of size (dim,dim), assuming that the pixels are in lexicographic order, and that the image is
        square.
        Returns an approximate position for each pixel, scaling the image to [0,1]^2.
        That is, the i-th pixel has the position [(i % n2) / (n2-1), (i // n2) / (n1-1)] in the domain [0,1]x[0,1].

        Parameters
        ---
        i :
            The index of the pixel, in lexicographic order. An integer between 0 and n1*n2-1.
            For example, i=n2-1 corresponds to the pixel at the upper right corner of the image.

        Returns
        ---
        array_like, shape (2, )
            The normalized position.
        """
        x_position = (i % self._n)
        y_position = (i // self._n)
        return np.array([x_position, y_position])
