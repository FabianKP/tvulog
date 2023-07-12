
import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional
import cvxpy as cp


MINTOL = 1e-15      # small tolerance so that ECOS does not abort prematurely


def solve_socp(a: csr_matrix, lb: np.array, ub: np.array, x_start: Optional[np.ndarray] = None,
               maxiters: int = 10, verbose: bool = True, tol: float = 1e-15) -> np.ndarray:
    """
    Solves the constrained total-variation minimization problem

    .. math::
        \\min_x ||D A x||_{1-2} \\text{ s. t. } \\ell \\leq x \\leq u,

    using a reformulation as SOCP. Here, :math:`A` is a differential operator that maps from :math:`\\mathbb R^n` to
    itself, and :math:`D` is a forward difference operator.

    Parameters
    ----------
    lb : shape (N, )
        The lower bound vector.
    ub : shape (N, )
        The upper bound vector.
    a : shape (N, N)
        The differential operator `A`.
    x_start: optional, shape (N, )
        Starting guess for the solution `x`.
    maxiters
        Maximum number of iterations for the interior point solver.
    tol
        Tolerance parameter for the solver. For the exact meaning, see the documentation of `CVXPY`.
    verbose
        Whether to print live progress to consolve.

    Returns
    -------
    x :
        The minimizer.
    """
    # Check input
    assert lb.shape == ub.shape
    # Rescale problem so that tube bounds are between -1 and 1.
    ndim = lb.ndim
    n = lb.size
    scale = ub.max()
    lb_scaled = lb.flatten() / scale
    ub_scaled = ub.flatten() / scale
    x_start = x_start.flatten() / scale
    # Set up the problem in CVXPY.
    x = cp.Variable(n)
    x.value = x_start
    l1l2norm = cp.mixed_norm(cp.reshape(a @ x, shape=(n, ndim), order="F"), p=2, q=1)
    target = cp.Minimize(l1l2norm)
    bounds = [x >= lb_scaled, x <= ub_scaled]
    cp_problem = cp.Problem(target, bounds)
    # Solve with ECOS.
    cp_problem.solve(verbose=verbose, solver=cp.ECOS, max_iters=maxiters, abstol=MINTOL, reltol=MINTOL, feastol=MINTOL,
                     abstol_inacc=tol, reltol_inacc=tol, feastol_inacc=100 * tol)
    x_min = x.value
    # Transfer back to original scale and return in correct shape.
    x_min *= scale
    x_min = x_min.reshape(lb.shape)
    return x_min
