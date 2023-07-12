
import numpy as np
from scipy.ndimage import laplace
import scipy.sparse as sspa
from scipy.sparse.linalg import LinearOperator
from typing import Sequence, Union

from ._differential_operator import DifferentialOperator


def laplacian1d(x: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian of the given vector, assuming Neumann boundary conditions.

    Parameters
    ----------
    x : shape (n, )
        Input vector.

    Returns
    -------
    y : shape (n, )
        The discrete Laplacian of `x`.
    """
    assert x.ndim == 1
    return laplace(x, mode="mirror")


class Laplacian1D(DifferentialOperator):
    """
    Implementation of the Laplace operator for one-dimensional signals as child of `DifferentialOperator`.
    It uses a central-difference approximation, assuming Neumann boundary conditions.
    """
    def __init__(self, size: int):
        """
        Parameters
        ----------
        size
            The size of the input signal.
        """
        shape = (size,)
        DifferentialOperator.__init__(self, x_shape=shape, y_shape=shape)
        self._csr_matrix = self._assemble_csr_matrix()

    @property
    def csr_matrix(self) -> Union[sspa.csr_matrix, Sequence[sspa.csr_matrix]]:
        """
        The sparse respresentation of the Laplacian.
        """
        return self._csr_matrix

    def _assemble_csr_matrix(self) -> sspa.csr_matrix:
        """
        Assembles the `csr_matrix` that represents the one-dimensional Laplacian.
        """
        x = []
        row_ind = []
        col_ind = []
        # Diagonal entries.
        x.extend(- 2 * np.ones(self.x_size))
        row_ind.extend(np.arange(self.x_size))
        col_ind.extend(np.arange(self.x_size))
        # Upper diagonal entries.
        upper_diagonal = np.ones(self.x_size - 1)
        upper_diagonal[0] = 2.
        x.extend(upper_diagonal)
        row_ind.extend(np.arange(self.x_size - 1))
        col_ind.extend(np.arange(self.x_size - 1) + 1)
        # Lower diagonal entries.
        lower_diagonal = np.ones(self.x_size - 1)
        lower_diagonal[-1] = 2.
        x.extend(lower_diagonal)
        row_ind.extend(np.arange(1, self.x_size))
        col_ind.extend(np.arange(1, self.x_size) - 1)
        csr_mat = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        return csr_mat

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Outputs the Laplacian applied to the given vector.
        """
        return self.csr_matrix.dot(x)

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        The discrete Laplacian is not self-adjoint because of the mirror-boundary conditions.
        """
        return self.csr_matrix.T.dot(y)