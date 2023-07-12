
from dataclasses import dataclass
import numpy as np


@dataclass
class CTVSolution:
    """
    Container for a solution of a `CTVProblem`.
    """
    x: np.ndarray           # The minimizer of the CTV problem.
    info: dict              # A dictionary with additional information, depending on solver.