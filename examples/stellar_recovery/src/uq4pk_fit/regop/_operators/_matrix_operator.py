
import numpy as np

from .._regularization_operator import RegularizationOperator


class MatrixOperator(RegularizationOperator):
    """
    Builds regularization operator from matrix.
    """
    def __init__(self, mat: np.ndarray):
        """
        Parameters
        ----------
        mat : shape (r, n)
            The matrix representation of the regularization operator.
        """
        RegularizationOperator.__init__(self, mat)

    def adj(self, v: np.ndarray) -> np.ndarray:
        return self._mat.T @ v

    def fwd(self, v: np.ndarray) -> np.ndarray:
        return self._mat @ v

    def inv(self, w: np.ndarray) -> np.ndarray:
        v = np.linalg.solve(self._mat, w)
        return v