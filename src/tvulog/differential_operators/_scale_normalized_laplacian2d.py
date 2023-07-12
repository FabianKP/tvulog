
import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sspa
from typing import Sequence

from ._differential_operator import DifferentialOperator
from ._laplacian2d import Laplacian2D


class ScaleNormalizedLaplacian2D(DifferentialOperator):
    """
    Implementation of scale-normalized Laplacian in one dimension as `DifferentialOperator`.
    If `x` is a scale-space representation of an image of shape `(m, n)`
    (i.e. a `(k, m, n)`-array :math:`[x_1, \\ldots, x_n]`), then the scale-normalized Laplacian is simply

    .. math::
        \\Delta^\\mathrm{norm} x = [\\Delta x_1, \\ldots, \\Delta x_k].

    Assuming that `x` is flattened in row-major order, the scale-normalized Laplacian can be represented by
    the `(kmn, kmn)`-matrix :math:`\\mathrm{diag}(t_1 D, \\ldots, t_k D)`,
    where `D` is the `(mn, mn)`-matrix that represents the two-dimensional Laplacian.
    """
    def __init__(self, m: int, n: int, sigmas: Sequence[float], width_to_height: float):
        """
        Parameters
        ----------
        m
            The number of rows of the underlying image.
        n
            The number of columns of the underlying image.
        sigmas
            The sigma-values for the scales-of-interest.
        width_to_height
            The width_to_height ratio , i.e. r = sigma_2 / sigma_1. Given `sigma`, the image is convolved along the
            x1-direction (vertical) with sigma, and along the x2-direction (horizontally) with sigma / r.
        """
        self._ssr_shape = (len(sigmas), m, n)
        self._sigmas = sigmas
        self._scales = [s * s for s in sigmas]  # The t-values.
        num_scales = len(sigmas)
        self._width_to_height = width_to_height
        shape = (num_scales, m, n)
        DifferentialOperator.__init__(self, x_shape=shape, y_shape=shape)
        self._csr_matrix = self._assemble_csr_matrix()

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the scale-normalized Laplacian to given input array.
        Parameters
        ----------
        arr : shape (k, n)

        Returns
        -------
        normlap_arr : shape (k, n)
        """
        assert arr.shape == self.x_shape
        normlap_arr = self.matvec(arr.flatten()).reshape(self.y_shape)
        return normlap_arr

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        The representation of the scale-normalized Laplacian as sparse matrix.
        """
        return self._csr_matrix

    def _assemble_csr_matrix(self) -> sspa.csr_matrix:
        """
        Assembles the sparse representation.
        The `csr_matrix` for the scale-normalized Laplacian is simply the block diagonal of the scaled Laplacians.
        """
        # Create `csr_matrix` for one-dimensional Laplacian.
        lap2d = Laplacian2D(m=self._x_shape[1], n=self._x_shape[2], width_to_height=self._width_to_height)
        lap2d_csrmat = lap2d.csr_matrix
        # Assemble block diagonal matrix of scaled one-dimensional Laplacians.
        blocks = [t * lap2d_csrmat for t in self._scales]
        scale_normalized_laplacian_csrmat = sspa.block_diag(mats=blocks, format="csr")
        return scale_normalized_laplacian_csrmat

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the scale-normalized Laplacian to the input.
        """
        return self._csr_matrix.dot(x)

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Applies the adjoint scale-normalized Laplacian to the input.
        """
        return self._csr_matrix.T.dot(y)