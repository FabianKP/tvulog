
import numpy as np

from .._regularization_operator import RegularizationOperator


class IdentityOperator(RegularizationOperator):
    """
    Corresponds to the identity operator :math:`I(v) = v`.
    """
    def __init__(self, dim):
        """
        Parameters
        ----------
        dim
            Dimension of the operator's domain.
        """
        mat = np.identity(dim)
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        return v

    def adj(self, v: np.ndarray) -> np.ndarray:
        return v

    def inv(self, w: np.ndarray) -> np.ndarray:
        return w