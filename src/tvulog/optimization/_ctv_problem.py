
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix


class CTVProblem:
    """
    Representation of the constrained total-variation minimization problem.

    .. math::
        \\min_x ||A x||_{1-2} \\text{ s. t. } \\ell \\leq x \\leq u.

    Here, `x` is an N-dimensional object and `A` is a flattened forward-difference-like operator that maps `x` to its
    N derivatives, but in flattened form.
    """
    def __init__(self, a: csr_matrix, lb: np.ndarray, ub: np.ndarray):
        """

        Parameters
        ----------
        a
            The differential operator.
        lb
            The lower bound in form of an N-dimensional numpy array.
        ub
            The upper bound. Must have the same shape as `lb`.
        """
        # Check consistency of dimensions.
        shape = lb.shape
        assert lb.shape == ub.shape
        self._a = a
        self._lb = lb
        self._ub = ub
        self._shape = shape

    @property
    def a(self) -> csr_matrix:
        """
        The differential operator.
        """
        return self._a

    @property
    def lb(self) -> np.ndarray:
        """
        The lower bound.
        """
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        """
        The upper bound.
        """
        return self._ub

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The original shape of `x`.
        """
        return self._shape
