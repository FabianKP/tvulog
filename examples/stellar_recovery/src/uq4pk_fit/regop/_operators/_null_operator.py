
import numpy as np

from .._regularization_operator import RegularizationOperator


class NullOperator(RegularizationOperator):
    """
    The null operator :math:`R(v) = 0`. Useful as stand-in if there is no regularization (e.g. in maximum-likelihood).
    """
    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim
            The dimension of the operator's domain.
        """
        mat = np.zeros((1, dim))
        RegularizationOperator.__init__(self, mat)
        self._rdim = 0

    def adj(self, v: np.ndarray):
        """
        Always returns 0.
        """
        return np.zeros((1,))

    def fwd(self, v: np.ndarray):
        """
        Always return 0.
        """
        return np.zeros((1,))

    def inv(self, w: np.ndarray):
        """
        Always return 0.
        """
        return np.zeros((1,))
