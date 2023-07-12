
import numpy as np

from ._ctv_problem import CTVProblem
from ._ctv_solution import CTVSolution


class CTVSolver:
    """
    Abstract base class for methods that solve the CTV-problem.
    """
    def solve_ctv_problem(self, problem: CTVProblem, x_start: np.ndarray) -> CTVSolution:
        raise NotImplementedError