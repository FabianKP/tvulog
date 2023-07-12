
import numpy as np
from typing import Sequence, Optional

from ..differential_operators import ScaleNormalizedForwardDifference1D, ScaleNormalizedForwardDifference2D, \
    ScaleNormalizedLaplacian1D, ScaleNormalizedLaplacian2D
from ._ctv_problem import CTVProblem


class TVULoGProblem(CTVProblem):
    """
    Object that represents the TVULoG optimization problem.
    """
    def __init__(self, lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float],
                 width_to_height: Optional[float] = None):
        """
        Parameters
        ----------
        lb : shape (k, n) or (k, m, n)
            Lower bound of scale-space tube.
        ub : shape (k, n) or (k, m, n)
            Upper bound of scale-space tube.
        sigmas
            Standard deviations of scale-space representation.
        width_to_height
            Ratio of blob-width to blob-height.
        """
        # Check for consistency.
        assert lb.shape == ub.shape
        # Initialize the main differential operator (denoted by "A" in the paper).
        signal_dim = lb.ndim - 1
        # It's easier to treat the 1D- and 2D-cases separately.
        if signal_dim == 1:
            x_size = lb.shape[1]
            nabla_norm = ScaleNormalizedForwardDifference1D(size=x_size, sigmas=sigmas)
            delta_norm = ScaleNormalizedLaplacian1D(size=x_size, sigmas=sigmas)
        elif signal_dim == 2:
            m, n = lb.shape[1:]
            assert width_to_height is not None, "For two-dimensional images, have to provide `width_to_height`."
            nabla_norm = ScaleNormalizedForwardDifference2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
            delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
        else:
            raise NotImplementedError("Only implemented for 1- or 2-dimensional signals.")
        # Get the csr-matrix.
        a = nabla_norm.csr_matrix @ delta_norm.csr_matrix
        # Call the constructor of the super class.
        CTVProblem.__init__(self, a=a, lb=lb, ub=ub)
        self.nabla_norm = nabla_norm
        self.delta_norm = delta_norm


class TVULoGProblemScaled(CTVProblem):
    """
    Object that represents the rescaled TVULoG optimization problem. It uses scaled bounds.
    """
    def __init__(self, lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float],
                 width_to_height: Optional[float] = None):
        """
        Parameters
        ----------
        lb : shape (k, n) or (k, m, n)
            Lower bound of scale-space tube.
        ub : shape (k, n) or (k, m, n)
            Upper bound of scale-space tube.
        sigmas
            Standard deviations of scale-space representation.
        width_to_height
            Ratio of blob-width to blob-height.
        """
        # rescale boundaries
        assert lb.shape == ub.shape
        signal_dim = lb.ndim - 1
        # Rescale the bounds.
        self._scales = np.array([s * s for s in sigmas])
        lb_scaled = self._rescale(lb)
        ub_scaled = self._rescale(ub)
        # Initialize the scale-normalized TV-of-LoG operators, but the normalized Laplacian is unscaled.
        one_vector = np.ones(len(sigmas))
        if signal_dim == 1:
            x_size = lb.shape[1]
            nabla_norm = ScaleNormalizedForwardDifference1D(size=x_size, sigmas=sigmas)
            delta_norm = ScaleNormalizedLaplacian1D(size=x_size, sigmas=one_vector)
        elif signal_dim == 2:
            m, n = lb.shape[1:]
            assert width_to_height is not None, "For two-dimensional images, have to provide `width_to_height`."
            nabla_norm = ScaleNormalizedForwardDifference2D(m=m, n=n, sigmas=sigmas, width_to_height=width_to_height)
            delta_norm = ScaleNormalizedLaplacian2D(m=m, n=n, sigmas=one_vector, width_to_height=width_to_height)
        else:
            raise NotImplementedError("Only implemented for 1- or 2-dimensional signals.")
        # Get the csr_matrix.
        a = nabla_norm.csr_matrix @ delta_norm.csr_matrix
        # Call the constructor of the super class.
        CTVProblem.__init__(self, a=a, lb=lb_scaled, ub=ub_scaled)
        self.nabla_norm = nabla_norm
        self.delta_norm = delta_norm

    def _rescale(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the scaling L_tilde(x, t) = t L(x, t) to a given scale space object L.
        """
        if arr.ndim == 2:
            arr = arr * self._scales[:, np.newaxis]
        elif arr.ndim == 3:
            arr = arr * self._scales[:, np.newaxis, np.newaxis]
        else:
            raise ValueError("Wrong dim.")
        return arr

    def _rescale_back(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the rescaling L_tilde(x, t) = t^{-1} L(x,t) to a given scale space object L.
        """
        if arr.ndim == 2:
            arr = arr / self._scales[:, np.newaxis]
        elif arr.ndim == 3:
            arr = arr / self._scales[:, np.newaxis, np.newaxis]
        else:
            raise ValueError("Wrong dim.")
        return arr
