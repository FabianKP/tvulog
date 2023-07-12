
import numpy as np

from src.uq4pk_src.observation_operator import ObservationOperator
from src.uq4pk_src.distribution_function import RandomGMM_DistributionFunction
from src.uq4pk_src.model_grids import MilesSSP

from ._forward_operator import ForwardOperator


class MassWeightedForwardOperator(ForwardOperator):
    """
    The forward operator that maps stellar distribution functions IN MASS WEIGHT to integrated light spectra.
    While this one is not used directly, the `LightWeightedForwardOperator` class depends on it, since it first
    creates a mass-weighted forward operator and then re-normalizes it.
    """
    def __init__(self, ssps: MilesSSP, theta: np.ndarray, dv: float = 10, do_log_resample: int = True,
                 hermite_order: int = 4, mask: np.ndarray = None):
        """
        Parameters
        ---
        ssps
            The SSPS grid.
        theta : shape (r, )
            Parameters for the Gauss-Hermite expansion.
        dv
            The dv-value used.
        do_log_resample
            If True, does resampling.
        hermite_order
            Order of the hermite expansion. Must satisfy `hermite_order = theta.size - 3`.
        mask : shape (k, )
            A Boolean array. The mask applied to the integrated light spectrum.
        """
        self._op = ObservationOperator(max_order_hermite=hermite_order,
                                       ssps=ssps,
                                       dv=dv,
                                       do_log_resample=do_log_resample)
        f_tmp = RandomGMM_DistributionFunction(modgrid=self._op.ssps).F
        self.m_f = f_tmp.shape[0]
        self.n_f = f_tmp.shape[1]
        self.dim_f = self.m_f * self.n_f
        self.dim_theta = 3 + hermite_order
        self._theta = theta
        self.grid = self._op.ssps.w
        self.modgrid = self._op.ssps
        # convert mask to indices and find measurement dimension
        y_tmp = self._op.evaluate(f_tmp, theta)
        self.dim_y_unmasked = y_tmp.size
        if mask is None:
            self.dim_y = y_tmp.size
            self.mask = np.full((self.dim_y,), True, dtype=bool)
        else:
            self.mask = mask
            y_tmp_masked = y_tmp[mask]
            self.dim_y = y_tmp_masked.size

    def fwd(self, f):
        y = self.fwd_unmasked(f)
        y_masked = y[self.mask]
        return y_masked

    def fwd_unmasked(self, f):
        f_im = np.reshape(f, (self.m_f, self.n_f))
        y = self._op.evaluate(f_im, self._theta)
        return y

    @property
    def mat(self):
        dydf = self.mat_unmasked
        dy_masked = dydf[self.mask, :]
        return dy_masked

    @property
    def mat_unmasked(self):
        g = self._jac_f()
        return g

    def _jac_f(self):
        """
        Computes the Jacobian of the non-linear version with respect to `f`. This gives the matrix representation
        of the unmasked forward operator.
        """
        # compute dGdf
        d_tildeS_ft_df = self._op.ssps.F_tilde_s * self._op.ssps.delta_zt
        # turn it into twodim array
        # d_tildeS_ft_df_twodim = d_tildeS_ft_df.reshape(d_tildeS_ft_df.shape[0],-1)
        V, sigma, h, M = self._op.unpack_Theta_v(self._theta)
        losvd_ft = self._op.losvd.evaluate_fourier_transform(self._op.H_coeffs,
                                                             V,
                                                             sigma,
                                                             h,
                                                             M,
                                                             self._op.omega)
        d_ybar_ft_df = np.einsum('i,ijk->ijk', losvd_ft, d_tildeS_ft_df)
        d_ybar_ft_df_twodim = d_ybar_ft_df.reshape(d_ybar_ft_df.shape[0], -1)
        d_ybar_df = np.apply_along_axis(np.fft.irfft, 0, d_ybar_ft_df_twodim, self._op.ssps.n_fft)
        return d_ybar_df
