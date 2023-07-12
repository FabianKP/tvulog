
from math import sqrt
import numpy as np
from typing import Optional

from ._bls_solve import bls_solve
from ..regop import RegularizationOperator


class NonnegativeLinearModel:
    """
    This class represents a truncated linear statistical model of the form::
        y = G f + error,
        error ~ N(0, Sigma_error),
        f ~ N(f_bar, Sigma_f / beta),
        f >= 0.
    """
    def __init__(self, y: np.ndarray, P_error: RegularizationOperator, G: np.ndarray,
                 P_f: RegularizationOperator, beta: float, f_bar: Optional[np.ndarray] = None,
                 scaling_factor: Optional[float] = 1.):
        """
        Parameters
        ----------
        y : shape (n, )
            The measurement data.
        P_error : `RegularizationOperator` with `P_error.dim = n`
            The regularization operator that corresponds to the error covariance `Sigma_error`.
        G : shape (n, p)
            The observation operator.
        P_f : RegularizationOperator with `P_f.dim = p`.
            The regularization operator that corresponds to the prior covariance `Sigma_f`.
        beta : float
            The regularization parameter.
        f_bar: shape (p, ), optional
            The prior mean for f. Defaults to numpy.zeros(p)
        scaling_factor : float, optional
            The expected scale of ||P_error(_y - Gf)||_2. Helps the optimization.
        """
        self._y = y
        self._P_error = P_error
        self._G = G
        self._P_f = P_f
        self._P_error = P_error
        self._beta = beta
        self._scaling_factor = scaling_factor
        if f_bar is None:
            self._f_bar = np.zeros(G.shape[1])
        else:
            self._f_bar = f_bar
        # Initialize lower bound.
        self._lb = np.zeros(G.shape[1])

    def fit(self):
        """
        Solves the optimization problem::
            minimize ||P_error(Gf - y)||_2^2 + beta * ||P_f(f - f_bar)||_2^2
            subject to 0 <= f.

        Returns
        -------
        f : shape (n, )
            The minimizer.
        """
        # Transform to CLS-scheme: min_x ||Hx - y2||_2^2 / scale
        h1 = self._P_error.fwd(self._G)
        y1 = self._P_error.fwd(self._y)
        h2 = sqrt(self._beta) * self._P_f.mat
        y2 = sqrt(self._beta) * self._P_f.fwd(self._f_bar)
        h = np.concatenate([h1, h2])
        y = np.concatenate([y1, y2])
        # Solve bounded least-squares problem.
        f_min = bls_solve(h=h, y=y, lb=self._lb, scale=self._scaling_factor)
        return f_min


