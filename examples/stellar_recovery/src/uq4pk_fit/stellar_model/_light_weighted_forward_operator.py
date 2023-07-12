
import numpy as np

from src.uq4pk_src.model_grids import MilesSSP

from ._forward_operator import ForwardOperator
from ._mass_weighted_forward_operator import MassWeightedForwardOperator


class LightWeightedForwardOperator(ForwardOperator):
    """
    The forward operator that maps stellar distribution functions IN LIGHT WEIGHT to integrated light spectra.
    """
    def __init__(self, theta: np.ndarray, ssps: MilesSSP, dv: float = 10, do_log_resample: bool = True,
                 hermite_order: int = 4, mask: np.ndarray = None):
        """
        Parameters
        ---
        theta : shape (r, )
            Parameters for the Gauss-Hermite expansion.
        ssps
            The SSPS grid.
        dv
            The dv-value used.
        do_log_resample
            If True, does resampling.
        hermite_order
            Order of the hermite expansion. Must satisfy `hermite_order = theta.size - 3`.
        mask : shape (k, )
            A Boolean array. The mask that is applied to the integrated light spectrum.
        """
        # Create a mass-weighted forward operator.
        self.ssps = ssps
        mass_weigthed_fwdop = MassWeightedForwardOperator(ssps=ssps, dv=dv, do_log_resample=do_log_resample,
                                                          hermite_order=hermite_order, mask=mask, theta=theta)
        if mask is None:
            dim_y = mass_weigthed_fwdop.dim_y
            mask = np.full((dim_y,), True, dtype=bool)
        self.dim_theta = theta.size
        self.theta_v = theta
        self.dim_y = mass_weigthed_fwdop.dim_y
        self.dim_y_unmasked = mass_weigthed_fwdop.dim_y_unmasked
        # Get the matrix representation at theta.
        self.m_f = mass_weigthed_fwdop.m_f
        self.n_f = mass_weigthed_fwdop.n_f
        x_um = mass_weigthed_fwdop.mat_unmasked
        x = x_um[mask, :]
        # normalize the sum of the columns.
        column_sums = np.sum(x, axis=0)
        # Divide by column sums.
        self._x_bar_unmasked = x_um / column_sums[np.newaxis, :]
        self._x_bar = self._x_bar_unmasked[mask, :]
        self.weights = column_sums
        self.mask = mask

    def fwd(self, f: np.ndarray) -> np.ndarray:
        """
        Maps given distribution function to spectrum, masked values are removed.
        """
        return self._x_bar @ f

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the matrix representation of the MASKED light-weighted forward operator.
        """
        return self._x_bar.copy()

    def fwd_unmasked(self, f: np.ndarray) -> np.ndarray:
        """
        Maps given distribution function to spectrum, masked values are NOT removed.
        """
        return self._x_bar_unmasked @ f

    @property
    def mat_unmasked(self) -> np.ndarray:
        """
        Returns the matrix representation of the UNMASKED light-weighted forward operator.
        """
        return self._x_bar_unmasked.copy()
