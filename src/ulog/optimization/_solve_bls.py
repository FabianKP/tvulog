
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import lsq_linear
from typing import Union

CTOL = 1e-15


def solve_bls(a: Union[np.array, LinearOperator], b: np.array, lb: np.array, ub: np.array) \
        -> np.ndarray:
    """
    Solves bounded least-squares problem

    .. math::
        \\min_x || A x - b ||_2^2 \\text{ s. t. } \\ell \\leq x \\leq u.

    Parameters
    ----------
    a : shape (M, N)
        The design matrix. It can either be provided as `numpy.ndarray` or as `scipy.sparse.linalg.LinearOperator`,
        in which case only the forward and adjoint action have to be implemented.
    b : shape (M, )
        The regressand. Its dimension M must equal `a.shape[0]`.
    lb : shape (N, )
        The lower bound :math:`\\ell`.
    ub : shape (N, )
        The upper bound :math:`u`.

    Returns
    -------
    x : shape (N, )
        The computed minimizer of the non-negative least-squares problem.
    """
    # Check input.
    if not (isinstance(a, np.ndarray) or isinstance(a, LinearOperator)):
        raise ValueError("The design matrix 'a' must given as `np.ndarray` or `scipy.sparse.linalg.LinearOperator`.")
    m, n = a.shape
    if b.shape != (m, ):
        raise ValueError(f"The regressand 'b' must have shape ({m}, ).")
    # Solve the optimization problem with `scipy.optimize.lsq_linear`.
    x = lsq_linear(A=a, b=b, bounds=(lb, ub)).x
    return x

