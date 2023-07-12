
import numpy as np
import scipy.sparse as sspa
from typing import Sequence

from ._differential_operator import DifferentialOperator
from ._laplacian1d import Laplacian1D


class ScaleNormalizedLaplacian1D(DifferentialOperator):
    """
    Implementation of scale-normalized Laplacian in one dimension as `DifferentialOperator`.
    If `x` is a scale-space representation of a one-dimensional signal of size `n`
    (i.e. a `(k, n)`-array :math:`[x_1, \\ldots, x_n]`), then the scale-normalized Laplacian is simply

    .. math::
        \\Delta^\\mathrm{norm} x = [\\Delta x_1, \\ldots, \\Delta x_k].

    Assuming that `x` is flattened in row-major order, the scale-normalized Laplacian can be represented by
    the `(kn, kn)`-matrix :math:`\\mathrm{diag}(t_1 D, \\ldots, t_k D)`,
    where `D` is the `(n, n)`-matrix that represents the one-dimensional Laplacian.
    """
    def __init__(self, size: int, sigmas: Sequence[float]):
        """
        Parameters
        ----------
        size
            The size of the underlying signal.
        sigmas
            The standard deviations for the scale-space representation.
        """
        self._ssr_shape = (len(sigmas), size)
        self._sigmas = sigmas
        self._scales = [s * s for s in sigmas]  # The t-values.
        num_scales = len(sigmas)
        x_shape = (num_scales, size)
        DifferentialOperator.__init__(self, x_shape=x_shape, y_shape=x_shape)
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
        lap1d = Laplacian1D(size=self.x_shape[1])
        lap1d_csrmat = lap1d.csr_matrix
        # Assemble block diagonal matrix of scaled one-dimensional Laplacians.
        blocks = [t * lap1d_csrmat for t in self._scales]
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
