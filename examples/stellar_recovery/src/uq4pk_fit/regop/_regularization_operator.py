
from copy import deepcopy
import numpy as np


class RegularizationOperator:
    """
    Abstract base class for regularization operators.
    Each child of RegularizationOperator must implement the methods `fwd` and `adj`, which give the forward and the
    adjoint action of the regularization operator.
    """
    def __init__(self, mat: np.ndarray):
        """
        Parameters
        ----------
        mat : shape (r, n)
            The matrix representation of the regularization operator.
        """
        if mat.ndim != 2:
            raise ValueError
        self._mat = deepcopy(mat)
        self._dim = mat.shape[1]
        self._rdim = mat.shape[0]

    @property
    def dim(self) -> int:
        """
        The dimension of the domain of the regularization operator.
        """
        return deepcopy(self._dim)

    @property
    def rdim(self) -> int:
        """
        The dimension of the codomain of the regularization operator.
        """
        return deepcopy(self._rdim)

    @property
    def mat(self) -> np.ndarray:
        """
        The matrix representation of the regularization operator. A matrix of shape `(self.rdim, self.dim)`.
        """
        return deepcopy(self._mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        Evaluates the forward action of the regularization operator.

        Parameters
        ---
        v : shape (self.dim, m)
            The array on which the regularization operator acts. If m > 1, then the operator is evaluated on each of
            the columns of `v` separately.

        Returns
        ---
        w : shape (self.rdim, m)
            The action on `v`. If m = 1, each column of `w` corresponds to the action on the corresponding column
            of `v`.
        """
        raise NotImplementedError

    def adj(self, w: np.ndarray) -> np.ndarray:
        """
        Evaluates the adjoint action of the regularization operator.

        w : shape (self.rdim, m)
            The array on which the adjoint acts. If m > 1, then the operator is evaluated on each of
            the columns of `w` separately.

        Returns
        ---
        v : shape (self.dim, m)
            The adjoint action on `w`. If m = 1, each column of `v` corresponds to the action on the corresponding
            column of `w`.
        """
        raise NotImplementedError

    def inv(self, w: np.ndarray) -> np.ndarray:
        """
        Evaluates the (pseudo-) inverse of the regularization operator on a vector `w`,
         i.e. returns the minimum-norm solution `v` of `R v = w`.

        Parameters
        ---
        w : shape (self.rdim, )

        Returns
        ---
        v : shape (self.dim, )
            The minimum-norm solution of `R v = w`.
        """
        raise NotImplementedError
