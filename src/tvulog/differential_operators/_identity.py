
import numpy as np
import scipy.sparse as sspa
from typing import Tuple

from ._differential_operator import DifferentialOperator


class Identity(DifferentialOperator):
    """
    N-dimensional identity operator
    """
    def __init__(self, x_shape: Tuple[int, ...]):
        DifferentialOperator.__init__(self, x_shape=x_shape, y_shape=x_shape)
        self._csr_matrix = sspa.identity(self.x_dim, format="csr")

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Simply returns `arr` without change.
        """
        assert arr.shape == self.x_shape
        return arr

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        The representation of the identity operator as `scipy.sparse.csr_matrix`.
        """
        return self._csr_matrix

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        assert x.size == self.x_size
        return x

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        assert y.size == self.y_size
        return y
