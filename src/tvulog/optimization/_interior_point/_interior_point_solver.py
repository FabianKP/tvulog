
import numpy as np
from time import time
from typing import Sequence

from .._ctv_problem import CTVProblem
from .._ctv_solver import CTVSolver
from .._ctv_solution import CTVSolution

from ._solve_socp import solve_socp
from ._solve_socp_timing import solve_socp_timing


class InteriorPointSolver(CTVSolver):
    """
    Solves a CTV problem using interior-point methods, by reformulating it as SOCP.
    """
    def __init__(self, max_iter: int = 100, verbose: bool = False, tol: float = 1e-15):
        """
        Parameters
        ----------
        max_iter
            Maximum number of interior-point iterations.
        verbose
            Toggles if information is printed to console.
        tol
            Tolerance parameter for convergence criterion of the solver. See documentation of `solve_socp`.
        """
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol

    def solve_ctv_problem(self, problem: CTVProblem, x_start: np.ndarray) -> CTVSolution:
        """
        Solves the CTV problem using the interior-point approach.

        Parameters
        ----------
        problem
            Instance of `CTVProblem`.
        x_start
            Initial guess of the solution.

        Returns
        -------
        ctv_solution
            An instance of `CTVProblem`.
        """
        t0 = time()
        x = solve_socp(a=problem.a, lb=problem.lb, ub=problem.ub, x_start=x_start, maxiters=self.max_iter,
                       verbose=self.verbose, tol=self.tol)
        t_socp = time() - t0
        info = {"time": t_socp}
        ctv_solution = CTVSolution(x=x, info=info)
        return ctv_solution

    def solve_with_timing(self, problem: CTVProblem, iterations: Sequence[int], x_start: np.ndarray,
                          large_tol: float = 1e16)\
            -> CTVSolution:
        """
        Solves the CTV-problem with the interior-point approach for different iteration numbers and times each run.

        Parameters
        ----------
        problem
            An instance of `CTVProblem`.
        iterations
            A sequence of the different iteration numbers.
        x_start
            Initial guess for the solver.
        large_tol
            A large tolerance for the solver. Set this as large as possible to prevent error messages.

        Returns
        -------
        solution
            An instance of `CTVSolution`, where the `info`-dictionary has a `trajectory` entry that contains a
            numpy array of shape (2, n) (with `n = len(iterations)`), where the first row corresponds to the
            computation times and the second row to the corresponding objective.
        """
        # Set tolerance very high so that ECOS does not raise an error.
        solution = solve_socp_timing(a=problem.a, lb=problem.lb, ub=problem.ub, iters=iterations, x_start=x_start,
                                     verbose=self.verbose, tol=large_tol)
        return solution
