"""
Contains class multiplied operator
"""

from copy import deepcopy
import numpy as np

from .._regularization_operator import RegularizationOperator


class MultipliedOperator(RegularizationOperator):
    """
    Implements a regularization operator that is created by right-multiplying a given regularization operator
    with an invertible matrix. That is, given a regularization operator :math:`R` and a matrix :math:`Q`, the
    new regularization operator corresponds to :math:`R Q`.
    """
    def __init__(self, regop: RegularizationOperator, q: np.ndarray):
        """
        Parameters
        ---
        regop
            The regularization operator :math:`R`.
        q
            The matrix :math:`Q` by which the regularization operator is multiplied. It must have shape (dim,m),
            where dim = :code:`regop.dim`.
        """
        self._op = deepcopy(regop)
        self._q = q.copy()
        mat = regop._mat @ q
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray):
        """
        Evaluates `R Q v`.
        """
        u = self._q @ v
        return self._op.fwd(u)

    def adj(self, w: np.ndarray):
        """
        Evaluates `(RQ).T v = Q.T R.T v`.
        """
        # (RQ)^* = Q.T R^*
        return self._q.T @ self._op.adj(w)

    def inv(self, w: np.ndarray):
        """
        Evaluates `(RQ)^(-1) w`. We first solve `R z = w` and afterwards `Q v = z`. Then `v = (RQ)^(-1) w`.
        """
        qv = self._op.inv(w)
        v = np.linalg.solve(self._q, qv)
        return v
