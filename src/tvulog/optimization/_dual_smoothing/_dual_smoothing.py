
from dataclasses import dataclass
from math import sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from time import time

from ...util._l1l2_norm import l1l2_norm
from .._ctv_solver import CTVSolver
from .._ctv_problem import CTVProblem
from .._ctv_solution import CTVSolution
from .._proximal_operators import block_ball_projection, truncation


class DualSmoothingFPG(CTVSolver):
    """
    Implements the dual-smoothing approach. The FPG method is used to solve the smoothed dual problem.
    """
    def __init__(self, stepsize: float = 1., mu: float = 1., max_iter: int = 100000):
        """

        Parameters
        ----------
        stepsize
            Step size for the FPG method.
        mu
            Smoothing parameter.
        max_iter
            Maximum number of FPG iterations.
        """
        self.stepsize = stepsize
        self.eta = mu
        self.max_iter = max_iter

    def solve_ctv_problem(self, problem: CTVProblem, x_start: np.ndarray) -> CTVSolution:
        """
        Solves the CTV problem using the dual smoothing approach.

        Parameters
        ----------
        problem
            The optimization problem needs to be provided as instance of `CTVProblem`.
        x_start
            Initial guess for optimizer.

        Returns
        -------
        solution
            An instance of `CTVSolution`.
        """
        a_op = aslinearoperator(problem.a)
        res = dual_smoothing_fpg(a=a_op, lb=problem.lb.flatten(), ub=problem.ub.flatten(), stepsize=self.stepsize,
                                 x_start=x_start.flatten(), mu=self.eta, max_iter=self.max_iter)
        info = {"trajectory": res.trajectory, "num_iterations": res.n_iter}
        x = res.x.reshape(problem.lb.shape)
        ctv_solution = CTVSolution(x=x, info=info)
        return ctv_solution


@dataclass
class DualSmoothingSolution:
    """
    Container object for the output of `dual_smoothing_fpg`.
    """
    x: np.array           # The optimizer.
    trajectory: np.array  # (2, n) array. First row are times for individual iterations, second row are corresponding objective values.
    n_iter: int           # Number of iterations.


def dual_smoothing_fpg(a: LinearOperator, lb: np.array, ub: np.array, stepsize: float, x_start: np.ndarray,
                       mu: float = 1., max_iter: int = 100000) -> DualSmoothingSolution:
    """
    Solves the optimization problem

    .. math::
        \\min_x ||Ax||_{1-2} \\text{ subject to } \\ell \\leq x \\leq u,

    using the fast dual proximal gradient descent method.

    Parameters
    ----------
    a : shape (dn, n)
        The matrix `A`.
    lb : shape (n, )
        The lower bound vector.
    ub : shape (n, )
        The upper bound vector.
    stepsize :
        The step size.
    mu :
        Smoothing parameter.
    x_start : shape (n, )
        Initial guess.
    max_iter :
        Maximum number of iterations.

    Returns
    -------
    solution : OptimizationSolution
        An `OptimizationSolution` instance.
    """
    # Check input.
    if not (isinstance(a, np.ndarray) or isinstance(a, LinearOperator)):
        raise ValueError("The design matrix 'a' must given as `np.ndarray` or `scipy.sparse.linalg.LinearOperator`.")
    m, n = a.shape
    assert m % n == 0
    dim = int(m / n)

    # --- INITIALIZATION
    # Initialize dual vector with zero (this is also done in Beck and Teboulle (2009)).
    p = np.zeros(dim * n)
    x = truncation(-a.T @ p / mu, lb=lb, ub=ub)
    # Set the step size.
    t_dual = mu * stepsize
    # Initialize momentum parameter `beta`.
    beta = 1.
    # Primal extrapolation will be denoted with y, dual extrapolation will be denoted with q.
    q = p
    # We count the total number of inner iterations since we break if it surpassed `max_iter`
    itcounter = 0
    t0 = time()
    # Initialize trajectory with objective at start.
    ax0 = a @ x_start
    obj_list = [_phi(ax=ax0, d=dim)]
    t_list = [0.]

    # --- ITERATION
    while itcounter < max_iter:
        itcounter += 1
        # Update p.
        dgrad = _nabla_psi_mu(atp=a.T @ q, a=a, mu=mu, lb=lb, ub=ub)
        p_old = p
        p = block_ball_projection(x=q + t_dual * dgrad, rho=1., d=dim)
        atp = a.T @ p
        # Update x.
        x = truncation(v=-atp / mu, lb=lb, ub=ub)
        ax = a @ x
        phi = _phi(ax=ax, d=dim)
        t_phi = time() - t0
        obj_list.append(phi)
        t_list.append(t_phi)
        # Extrapolate
        beta_old = beta
        beta = 0.5 * (1 + sqrt(1 + 4 * beta ** 2))
        q = p + (beta_old - 1) / beta * (p - p_old)

    # Post-process and return `DualSmoothingSolution` object.
    trajectory = np.array([t_list, obj_list])
    solution = DualSmoothingSolution(x=x, trajectory=trajectory, n_iter=itcounter)
    return solution


def _phi(ax: np.array, d: int):
    """
    Evaluates the primal objective :math:`\Phi(x) = ||A x||_{1-2}`.

    Parameters
    ----------
    ax : shape (d, n)
        The matrix `Ax`.
    d
        The underlying signal dimension.

    Returns
    -------
    y
        The value :math:`\Phi(x)`.
    """
    return l1l2_norm(ax, d)


def _nabla_psi_mu(atp: np.array, a: LinearOperator, mu: float, lb: np.array, ub: np.array) -> np.array:
    """
    Returns the gradient of the smoothed dual objective:

    .. math::
        \\nabla \\Psi_{\\mu}(p) = \\mu A P_{[\\ell, u]}(\\frac{1}{\\mu} A^\\top p).

    Parameters
    ----------
    atp : shape (n, )
        The value of :math:`A^\top p`.
    mu
        The smoothing parameter.
    lb : shape (n, )
        The lower bound vector.
    ub : shape (n, )
        The upper bound vector.

    Returns
    -------
    nabla_Psi_p : shape (n, )
        The gradient of :math:`\Psi_\mu` evaluated at `p`.
    """
    nabla_Psi_p = a @ _inner_nabla_psi_mu(y=atp, mu=mu, lb=lb, ub=ub)
    return nabla_Psi_p


def _inner_nabla_psi_mu(y: np.array, mu: float, lb: np.array, ub: np.array) -> float:
    """
    Evaluates the scaled projection.

    .. math::
        \\nabla \\psi_\\eta(p) = P_{[\\ell, u]}(-\\frac{1}{\\eta} y)
    """
    y_eta = y / mu
    nabla_psi_y = truncation(v=-y_eta, lb=lb, ub=ub)
    return nabla_psi_y