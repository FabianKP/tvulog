
import numpy as np
import scipy.sparse as sspa
from typing import Sequence, Tuple

from ._differential_operator import DifferentialOperator
from ._forward_difference import ForwardDifference


class ScaleNormalizedForwardDifference2D(DifferentialOperator):
    """
    Scale-normalized forward difference operator for 2D-signals (images), implemented as child of
    `DifferentialOperator`.
    """
    def __init__(self, m: int, n: int, sigmas: Sequence[float], width_to_height: float):
        """
        Parameters
        ----------
        m
            Number of image rows.
        n
            Number of image columns.
        sigmas
            Standard deviations for the scale-space representation.
        width_to_height
            Weighting between horizontal and vertical direction.
        """
        num_sigmas = len(sigmas)
        shape = (num_sigmas, m, n)
        y_shape = (3, num_sigmas, m, n)
        self._sigmas = sigmas
        self._scales = [s * s for s in sigmas]
        self._width_to_height = width_to_height
        DifferentialOperator.__init__(self, x_shape=shape, y_shape=y_shape)
        self._csr_matrices = self._assemble_csr_matrices()
        self._csr_matrix = sspa.vstack(self._csr_matrices)

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the differential operator to a given array.

        Parameters
        ----------
        arr : shape self.x_shape
            The input array.

        Returns
        -------
        shape self.y_shape
            The output as array.
        """
        assert arr.shape == self.x_shape
        fdiff_arr = self.matvec(arr.flatten()).reshape(self.y_shape, order="C")
        return fdiff_arr

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        Returns matrix representation of forward difference operator along all axes. The output is a single
        `scipy.sparse.csr_matrix` instance.
        """
        return self._csr_matrix

    @property
    def csr_matrix_list(self) -> Tuple[sspa.csr_matrix, ...]:
        """
        Matrix representations of the scale-normalized forward difference operator along all axes.
        The output is a list, where each element is a `scipy.sparse.csr_matrix` instance, each corresponding to the
        derivative along a single direction.
        """
        return self._csr_matrices

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Applies differential operator to given flattened array.
        """
        assert x.size == self.x_size
        return self._csr_matrix @ x

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Implements the adjoint.
        """
        assert y.size == self.y_size
        return self._csr_matrix.T @ y

    def _assemble_csr_matrices(self) -> Tuple[sspa.csr_matrix, sspa.csr_matrix, sspa.csr_matrix]:
        """
        Assembles the csr matrices.
        """
        # Create unnormalized forward difference operator.
        unnorm_fdiff = ForwardDifference(shape=self.x_shape, width_to_height=self._width_to_height)
        # Get the three CSR matrices.
        scale_diff, space_diff1, space_diff2 = unnorm_fdiff.csr_matrix_list
        # Since we assume that the input is flattened in row-major order, the first n columns (n=x_size) need to be
        # multiplied with sigma_1, the next n columns with sigma_2, and so on.
        scales = np.square(self._sigmas)
        b = scales[1] / scales[0]
        scale_factor = 1 / (b - 1.)
        scale_diff = scale_factor * scale_diff
        sigma_weights = np.concatenate([sigma * np.ones(self.x_shape[1] * self.x_shape[2]) for sigma in self._sigmas])
        normalized_space_diff1 = space_diff1 * sspa.diags(sigma_weights)
        normalized_space_diff2 = space_diff2 * sspa.diags(sigma_weights)
        return scale_diff, normalized_space_diff1, normalized_space_diff2