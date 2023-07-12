
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from typing import Union

from ...util import l1l2_norm
from .._ctv_solver import CTVSolver
from .._ctv_problem import CTVProblem
from .._ctv_solution import CTVSolution
from ._fast_projected_gradient import fast_projected_gradient
from ._primal_smoothing_solution import PrimalSmoothingSolution


class PrimalSmoothingFPG(CTVSolver):
    """
    Implementation of the primal smoothing approach. Uses the FPG method to solve the minimization problem.
    """
    def __init__(self, stepsize: float = 1., mu: float = 1., max_iter: int = 100000):
        """

        Parameters
        ----------
        stepsize
            Step size for the FPG method.
        mu
            Smoothing parameter for the Nesterov approximation.
        max_iter
            Maximum number of FPG iterations.
        """
        self.stepsize = stepsize
        self.mu = mu
        self.max_iter = max_iter

    def solve_ctv_problem(self, problem: CTVProblem, x_start: np.ndarray) -> CTVSolution:
        """
        Solves the CTV problem using the FPG method with primal smoothing.

        Parameters
        ----------
        problem
            The optimization problem as instance of `CTVProblem`.
        x_start
            The initial guess.

        Returns
        -------
        ctv_solution
            Instance of `CTVSolution`.
        """
        a_op = aslinearoperator(problem.a)
        res = primal_smoothing_fpg(a=a_op, mu=self.mu, lb=problem.lb.flatten(), ub=problem.ub.flatten(),
                                   x0=x_start.flatten(), stepsize=self.stepsize, max_iter=self.max_iter)
        info = {"trajectory": res.trajectory, "num_iterations": res.n_iter}
        x = res.x.reshape(problem.shape)
        ctv_solution = CTVSolution(x=x, info=info)
        return ctv_solution


def primal_smoothing_fpg(a: Union[np.array, LinearOperator], mu: float, lb: np.array, ub: np.array,
                         x0: np.array = None, stepsize: float = 1., max_iter: int = 30000)\
        -> PrimalSmoothingSolution:
    """
    Solves the optimization problem
    .. math::
        \\min_x \\Phi_\\mu(Ax) \\text{ subject to } \\ell \\leq x \\leq u,
    where :math:`\\Phi_\\mu` is a smoothed l1-l2-norm.
    Parameters
    ----------
    a : shape (dn, n)
        The matrix `A`.
    mu :
        The smoothing parameter :math:`\mu`.
    lb : shape (n, )
        The lower bound vector.
    ub : shape (n, )
        The upper bound vector.
    x0 : shape (n, )
        Optional initial guess.
    stepsize :
        Maximum eigenvalue of `A.T A`.
    max_iter :
        Maximum number of APGD iterations.

    Returns
    -------
    solution :
        An `OptimizationSolution` instance.
    """
    # Check input.
    if not (isinstance(a, np.ndarray) or isinstance(a, LinearOperator)):
        raise ValueError("The design matrix 'a' must given as `np.ndarray` or `scipy.sparse.linalg.LinearOperator`.")
    m, n = a.shape
    assert mu > 0.
    assert m % n == 0
    d = int(m / n)
    # The Lipschitz constant of nabla Psi_mu is bounded by ||A.T A||_2 / mu.
    stepsize = stepsize * mu

    # Define smoothed objective.
    def objective(x: np.array) -> float:
        return _phi(a @ x, mu=mu, d=d)

    # Define gradient of smoothed objective.
    def objective_gradient(x: np.array) -> float:
        return a.T @ _phi_gradient(a @ x, mu=mu, d=d)

    # Target is the unsmoothed objective.
    def target(x: np.array) -> float:
        return l1l2_norm(a @ x, d=d)

    # Solve the optimization problem with APGD.
    solution = fast_projected_gradient(obj=objective, gradient=objective_gradient, lb=lb, ub=ub, x0=x0,
                                       stepsize=stepsize, max_iter=max_iter, target=target)
    return solution


def _phi(x: np.array, mu: float, d: int) -> float:
    """
    Evaluates the function

    .. math::
        \\Phi_\\mu(x) := \\sum_{i=1}^n \\psi_\mu(x),

    where :math:`X` is the (n,d)-array that results from reshaping `x` in column-major order.

    Parameters
    ----------
    x : shape (dn, )
        Vector at which function should be evaluated.
    mu
        Smoothing parameter.
    d
        Underlying dimension.

    Returns
    -------
    y :
        The value of :math:`\Phi_\mu(x)`.
    """
    m = x.size
    assert (m % d == 0) and (x.shape == (m, ))
    # Reshape x into (n, d) array, column-major ordering.
    x = np.reshape(x, (-1, d), order="F")
    # Compute psi_mu(x)
    small_psis = _small_phi(x, mu, d)
    # Sum over and return.
    y = np.sum(small_psis)
    return float(y)


def _phi_gradient(x: np.array, mu: float, d: int) -> np.array:
    """
    Evaluates the gradient of :math:`\\Phi_\\mu`, i.e.

    .. math::
        \\nabla \\Phi_\\mu(x)[i] := \\frac{X[i]}{\\sqrt{||X[i]||^2 + \\mu^2}}.

    Parameters
    ----------
    x : shape (dn, )
        Vector at which function should be evaluated.
    mu
        Smoothing parameter.
    d
        Underlying dimension.

    Returns
    -------
    g : shape (dn, )
        The gradient :math:`\\nabla \\Phi_\\mu(x)`.
    """
    m = x.size
    assert (m % d == 0) and (x.shape == (m, ))
    # Reshape x into (n, d) array.
    x = np.reshape(x, (-1, d), order="F")
    # Evaluate inner gradients.
    g = _small_psi_gradient(x=x, mu=mu, d=d)
    # Flatten in column-major order.
    g = g.flatten(order="F")
    return g


def _small_phi(x: np.array, mu: float, d: int) -> np.array:
    """
    Evaluates

    .. math::
        \\phi_\\mu(x_i) = |x_i|, \\text{ if } |x_i| \\geq \\mu, \\
        \\phi_\\mu(x_i) = \\frac{|x_i|^2}{2 \\mu} + \\frac{\\mu}{2}, \text{ otherwise},

    where `x_i` are the `d`-dimensional components of `x`.

    Parameters
    ---
    x : shape (dn, )
        Vector at which function should be evaluated.
    mu
        Smoothing parameter.
    d
        Underlying dimension.

    Returns
    ---
    y : shape (n, )
        Vector of function values :math:`\phi_\mu(x_i)`.
    """
    assert x.ndim == 2 and x.shape[1] == d
    y = np.linalg.norm(x, axis=1)
    smaller_mu = np.where(y < mu)
    x_quad = 0.5 * (np.square(y) / mu + mu)
    y[smaller_mu] = x_quad[smaller_mu]
    return y


def _small_psi_gradient(x: np.array, mu: float, d: int) -> np.array:
    """
    Evaluates the gradient of :math:`\\phi_\\mu`.

    .. math::
        \\nabla \\phi_\\mu(x) = \\frac{x}{||x||}, \\text{ if } |x| \\geq \\mu, \\
        \\nabla \\phi_\mu(x) = \\frac{x}{\\mu}, \\text{ otherwise}.

    Parameters
    ---
    x : shape (dn, )
        Vector at which function should be evaluated.
    mu
        Smoothing parameter.
    d
        Underlying dimension.

    Returns
    ---
    g : shape (n, d)
        Array of gradient-values :math:`\nabla \phi_\mu(x_i)` (each row is a gradient).
    """
    assert x.ndim == 2 and x.shape[1] == d
    x_norms = np.linalg.norm(x, axis=1)
    larger_mu = np.where(x_norms >= mu)
    g = x / mu
    g[larger_mu] = x[larger_mu, :] / x_norms[larger_mu, np.newaxis]
    return g
