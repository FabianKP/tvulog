
from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence

from ._tv_ulog_problem import TVULoGProblem, TVULoGProblemScaled
from ..differential_operators import scale_normalized_total_variation
from ._interior_point import InteriorPointSolver


@dataclass
class TVULoGProblemSolution:
    minimizer: np.ndarray   # Solution of TV-ULoG optimization problem.
    normlap: np.ndarray     # Its normalized Laplacian.
    normalized_tv: float    # (Scale-normalized) total variation of the normalized Laplacian.


def tv_ulog(lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float], width_to_height: Optional[float] = None,
            scaled: bool = True, options: dict = None) -> TVULoGProblemSolution:
    """
    Solves the TV-ULoG optimization problem for 1D-signals or 2D-signals (images).

    Parameters
    ----------
    lb : shape (k, n) or (k, m, n)
        The lower bound of the scale-space tube.
    ub : shape (k, n) or (k, m, n)
        The upper bound of the scale-space tube.
    sigmas
        The standard deviations of the scale-space representation.
    width_to_height
        The width-to-height ratio.
    scaled
        Uses a transformation to achieve better conditioning of the problem. If the bounds are rescaled like
        `lb[k,i] = t[k] * lb[k, i], ub[k, i] = t[k] * ub[k, i]` (in the 1D case, 2D case is analog),
        then the normalized Laplacian can be replaced with the unnormalized Laplacian.
        However, for the interior-point method the difference in the results is not significant.
    options
        A dictionary with additional options for the solver
        - verbose: Toggles if information is printed to console.
        - max_iter: Maximum number of iterations for solver.
        - tol: Tolerance parameter for solver.
        - x_start: Initial guess for solver.

    Returns
    -------
    solution
        Instance of `TVULoGProblemSolution`.
    """
    if options is None:
        options = {}
    # Uses separate implementations for scaled and unscaled case.
    # This could also be unified to reduce duplication, but this way it's easier to verify.
    if scaled:
        return _tv_ulog_scaled(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height, options=options)
    else:
        return _tv_ulog_unscaled(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height, options=options)


def _tv_ulog_unscaled(lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float], width_to_height: Optional[float] = None,
                      options: dict = None) -> TVULoGProblemSolution:
    # Create TV-ULoG Problem
    problem = TVULoGProblem(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height)
    # Solve it with SOCP solver.
    verbose = options.setdefault("verbose", True)
    max_iter = options.setdefault("max_iter", 100)
    tol = options.setdefault("tol", 1e-4)
    x_start_default = 0.5 * (lb + ub)
    x_start = options.setdefault("x_start", x_start_default)
    socp_solver = InteriorPointSolver(max_iter=max_iter, tol=tol, verbose=verbose)
    ctv_solution = socp_solver.solve_ctv_problem(problem=problem, x_start=x_start)
    # Extract minimizer from CTVSolution.
    blanket = ctv_solution.x
    # Compute scale-normalized Laplacian.
    normlap = problem.delta_norm.fwd_arr(blanket)
    # Check that blanket is in the bounds.
    bound_err = np.max((blanket - ub).clip(min=0.)) + np.max((lb - blanket).clip(min=0.))
    print(f"Relative bound error = {bound_err / np.max(blanket)}.")
    # Compute target quantity.
    normalized_tv = scale_normalized_total_variation(normlap, sigmas, width_to_height)
    print(f"Normalized TV: {normalized_tv}.")
    return TVULoGProblemSolution(blanket, normlap, normalized_tv)


def _tv_ulog_scaled(lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float], width_to_height: Optional[float] = None,
                    options: dict = None):
    """
    Version of TV-ULoG that uses a better conditioned formulation of the optimization problem.
    """
    # Create TV-ULoG Problem
    problem = TVULoGProblemScaled(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height)
    # Solve it with SOCP solver.
    verbose = options.setdefault("verbose", True)
    max_iter = options.setdefault("max_iter", 100)
    tol = options.setdefault("tol", 1e-4)
    x_start = 0.5 * (lb + ub)
    x_start = problem._rescale(x_start)
    socp_solver = InteriorPointSolver(max_iter=max_iter, tol=tol, verbose=verbose)
    ctv_solution = socp_solver.solve_ctv_problem(problem=problem, x_start=x_start)
    # Extract minimizer from CTVSolution.
    blanket_scaled = ctv_solution.x
    # Rescale blanket back to original scale.
    blanket = problem._rescale_back(blanket_scaled)
    # Compute scale-normalized Laplacian
    problem_unscaled = TVULoGProblem(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height)
    normlap = problem_unscaled.delta_norm.fwd_arr(blanket)
    # Check that blanket is in the bounds.
    bound_err = np.max((blanket - ub).clip(min=0.)) + np.max((lb - blanket).clip(min=0.))
    print(f"Relative bound error = {bound_err / np.max(blanket)}.")
    # Compute target quantity.
    normalized_tv = scale_normalized_total_variation(normlap, sigmas, width_to_height)
    print(f"Normalized TV: {normalized_tv}.")
    return TVULoGProblemSolution(blanket, normlap, normalized_tv)
