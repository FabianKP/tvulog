
import numpy as np
from typing import Union

from .._regularization_operator import RegularizationOperator


class DiagonalOperator(RegularizationOperator):
    """
    Tool to form diagonal regularization operators of the form :math:`R = \mathrm{diag}(s_1, \ldots, s_n)`.
    """
    def __init__(self, dim, s: Union[float, np.ndarray]):
        """
        If s is a float, the resulting diagonal operator will be :math:`R = \mathrm{diag}(s, \ldots, s)`.
        If s is a vector, the resulting diagonal operator will be :math:`R = \mathrm{diag}(s_1, \ldots, s_n)`.

        Parameters
        ---
        dim
            The dimension `n`.
        s
            If `s` is a float, the operator is
            given by `R = s * np.identity(dim)`, otherwise if `s` is an array of shape `(dim, )`, the operator `R`
            is the diagonal matrix with diagonal given by `s`.
        """
        is_float = isinstance(s, float)
        if is_float:
            is_positive = s > 0
        else:
            is_positive = np.all(s > 0)
        if not is_positive:
            raise ValueError("s must be positive.")
        if not is_float:
            if s.size != dim or s.ndim != 1:
                raise ValueError("s must be float or vector of shape (dim,).")
        self._isfloat = is_float
        self._s = s
        mat = s * np.identity(dim)
        RegularizationOperator.__init__(self, mat)

    def adj(self, v: np.ndarray):
        """
        Adjoint action of diagonal operator. Same as forward action since diagonal operators are self-adjoint.
        """
        return self.fwd(v)

    def fwd(self, v: np.ndarray):
        """
        The forward action of the diagonal operator on `v`.
        """
        if v.ndim == 1:
            u = self._s * v
        else:
            # matrix case
            if self._isfloat:
                u = self._s * v
            else:
                u = self._s[:, np.newaxis] * v
        return u

    def inv(self, w: np.ndarray) -> np.ndarray:
        """
        The pseudo-inverse evaluated at `w`::

            R = s Id => R^(-1) = s^(-1) Id.
        """
        v = np.divide(w, self._s)
        return v
