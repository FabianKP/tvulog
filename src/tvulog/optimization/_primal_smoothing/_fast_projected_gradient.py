

from math import sqrt
import numpy as np
from time import time
from typing import Optional, Union

from .._proximal_operators import truncation
from ._primal_smoothing_solution import PrimalSmoothingSolution


def fast_projected_gradient(obj: callable, gradient: callable,  lb: np.array, ub: np.array,
                            x0: Optional[np.array] = None, stepsize: float = 1., max_iter: Optional[int] = 10000,
                            alpha_start: float = 0.9, target: callable = None) -> PrimalSmoothingSolution:
    """
    Solves the minimization problem
    .. math::
        \\min_x f(x) \\quad \\text{subject to} \\quad \\ell \\leq x \\leq u,
    using the fast projected gradient (FPG) method.

    Parameters
    ----------
    obj :
        The objective function.
    gradient :
        The gradient of the objective function.
    lb : shape (N, )
        The lower bound.
    ub : shape (N, )
        The upper bound.
    x0 : shape (N, )
        An optional initial guess.
    stepsize :
        The step size of the iteration.
    max_iter :
        Maximum number of iterations.
    alpha_start :
        Starting value for momentum variable.
    target : callable
        If provided, the target is evaluated every iteration. The values are stored in the solution as `target_traj`.

    Returns
    -------
    sol :
        An object of type `PrimalSmoothingSolution`.
    """
    # Check input.
    n = lb.size
    assert lb.shape == ub.shape == (n, )
    assert np.all(lb <= ub)
    x0 = _set_initial_value(x0, lb, ub)
    if not (isinstance(max_iter, int) and max_iter >= 1):
        raise ValueError(f"'max_iter' must be an integer greater or equal 1.")

    # Initialize variables
    y = x0
    x_old = x0
    x = x0
    alpha_old = alpha_start
    t = stepsize
    # Initialize lists for trajectory.
    obj_traj = [obj(x0)]
    target_list = [target(x0)]
    time_list = [0.]
    t0 = time()
    n_iter = 0
    for k in range(max_iter):
        n_iter += 1
        x = truncation(y - t * gradient(y), lb, ub)
        obj_x = obj(x)
        obj_traj.append(obj_x)
        if not np.isfinite(obj_x):
            raise RuntimeError("Infinite objective. Try to re-run with decreased step size.")
        # Evaluate target if exists.
        time_list.append(time() - t0)
        if target is not None:
            target_x = target(x)
            target_list.append(target_x)
        alpha = 0.5 * (1 + sqrt(1 + 4 * alpha_old ** 2))
        beta = (alpha_old - 1) / alpha
        y = x + beta * (x - x_old)
        x_old = x
        alpha_old = alpha

    trajectory = np.array([time_list, target_list])
    solution = PrimalSmoothingSolution(x=x, trajectory=trajectory, n_iter=n_iter)
    return solution


def _set_initial_value(x_start: Union[np.array, None], lb: np.array, ub: np.array) -> np.array:
    """
    Checks x_start and sets it to default if not provided.
    """
    if x_start is None:
        x_start = np.zeros_like(lb)
        lb_infinite_ub_finite = np.where((lb == -np.infty) & (np.isfinite(ub)))
        x_start[lb_infinite_ub_finite] = ub[lb_infinite_ub_finite]
        ub_infinite_lb_finite = np.where((ub == np.infty) & (np.isfinite(lb)))
        x_start[ub_infinite_lb_finite] = lb[ub_infinite_lb_finite]
        both_finite = np.where((np.isfinite(lb)) & (np.isfinite(ub)))
        x_start[both_finite] = 0.5 * (lb + ub)[both_finite]
    return x_start
