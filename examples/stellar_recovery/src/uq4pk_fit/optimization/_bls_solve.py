
from math import sqrt
import numpy as np
from typing import Optional

from scipy.optimize import lsq_linear


def bls_solve(h: np.ndarray, y: np.ndarray, scale: float = 1., lb: Optional[np.ndarray] = None,
              ub: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solves bounded least-squares problem::

        minimize ||H x - y ||_2^2
        subject to lb <= x <= ub.

    Parameters
    ----------
    h   : shape (m, n)
        The design matrix.
    y : shape (m, n)
        The regressand.
    scale
        A strictly positive scaling factor. It does not change the optimization problem, but a good choice can help
        the solver. A good default choice is for example `scale = y.size`.
    lb : shape (n, )
        The lower bound. Can contain values equal to `-np.inf` to indicate no lower bound on that variable.
    ub : shape (n, )
        The upper bound. Can contain values equal to `np.inf` to indicate no lower bound on that variable.

    Returns
    -------
    x_min
        The minimizer of the bounded least-squares problem.
    """
    m, n = h.shape
    # Set the bound constraints if they are None.
    if lb is None:
        lb = - np.infty * np.ones(n)
    if ub is None:
        ub = np.infty * np.ones(n)
    # Check consistency of the input.
    assert np.all(lb <= ub)
    assert y.shape == (m, )
    assert lb.shape == ub.shape == (n, )
    assert scale > 0.
    # Rescale h and y.
    h_scaled = h / sqrt(scale)
    y_scaled = y / sqrt(scale)
    # Solve the problem with `scipy.optimize.lsq_linear`.
    x_min = lsq_linear(A=h_scaled, b=y_scaled, bounds=(lb, ub)).x
    return x_min
