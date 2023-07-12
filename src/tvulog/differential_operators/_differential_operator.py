
from functools import reduce
import numpy as np
import scipy.sparse as sspa
from scipy.sparse.linalg import LinearOperator
from typing import Sequence, Union, Tuple


class DifferentialOperator(LinearOperator):
    """
    Abstract base class for differential operators.
    """
    def __init__(self, x_shape: Tuple[int, ...], y_shape: Tuple[int, ...]):
        assert len(x_shape) >= 1
        assert len(y_shape) >= 1
        x_size = reduce(lambda x, y: x * y, x_shape)
        y_size = reduce(lambda x, y: x * y, y_shape)
        x_dim = len(x_shape)
        y_dim = len(y_shape)
        self._x_dim = x_dim
        self._x_size = x_size
        self._x_shape = x_shape
        self._y_dim = y_dim
        self._y_size = y_size
        self._y_shape = y_shape
        LinearOperator.__init__(self, shape=(y_size, x_size), dtype=np.float64)

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the differential operator to a given array.

        Parameters
        ----------
        arr : shape `self.x_shape`
            The input array.

        Returns
        -------
        out : shape `self.y_shape`
            The output as array.
        """
        raise NotImplementedError

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        Returns the differential operator represented as `scipy.sparse.csr_matrix`. If the differential operator is
        direction-dependent, returns multiple arrays corresponding to the derivative along each direction.
        """
        raise NotImplementedError

    @property
    def x_size(self) -> int:
        """
        # The overall dimension of the domain.
        """
        return self._x_size

    @property
    def x_dim(self) -> int:
        """
        The number of dimensions of the domain.
        """
        return self._x_dim

    @property
    def x_shape(self) -> Tuple[int, ...]:
        """
        The shape of the signal domain.
        """
        return self._x_shape

    @property
    def y_size(self) -> int:
        """
        The size of the codomain.
        """
        return self._y_size

    @property
    def y_dim(self) -> int:
        """
        The dimension of the codomain.
        """
        return self._y_dim

    @property
    def y_shape(self) -> Tuple[int, ...]:
        """
        The shape of the codomain.
        """
        return self._y_shape

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Applies differential operator to given flattened array.

        Parameters
        ----------
        x : shape (self.x_size, )
            The array, flattened in row-major order.

        Returns
        -------
        y : shape (self.y_size, )
            The output, flattened in row-major order.
        """
        raise NotImplementedError

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Applies the adjoint differential operator to flattened input.

        Parameters
        ----------
        y : shape (self.y_dim, )

        Returns
        -------
        x : shape (self.x_dim, )
        """
        raise NotImplementedError
