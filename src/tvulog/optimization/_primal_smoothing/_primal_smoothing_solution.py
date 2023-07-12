
from dataclasses import dataclass
import numpy as np


@dataclass
class PrimalSmoothingSolution:
    """
    Container object for the result of `primal_smoothing_fpg`.
    """
    x: np.array           # The optimizer.
    trajectory: np.array  # (2, n) array. First row are times for individual iterations, second row are corresponding objective values.
    n_iter: int           # Number of iterations.