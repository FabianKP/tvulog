import numpy as np
from . import model_grids, losvds

class ObservationOperator:
    """Observation operator for stellar population-kinematical modelling.

    The method ``evaluate`` can be used to evaluate an observed signal given an
    input distributuion function ``f`` and parameters ``Theta_v`` of a
    Gauss-Hermite LOSVD. This class itself requires three parameters to be
    instantiated - the defaults provided below are realistic and may be left
    untouched.

    Parameters
    ----------
    ssps : model_grids.MilesSSP
        Object holding the SSP templates
    dv : float
        velocity scale in km/s
    max_order_hermite : int
        maximum desired order of Hermite polynomial

    """

    def __init__(self,
                 ssps=model_grids.MilesSSP(),
                 dv=10.,
                 max_order_hermite=4,
                 do_log_resample=True):
        # interpolate SSPS at logarithmically spaced lambda and calculate FTs
        if do_log_resample:
            ssps.logarithmically_resample(dv=dv)
        ssps.calculate_fourier_transform(pad=False)
        # reshape the Fourier transform of the SSPs i.e. FT(tilde{s})
        ssps.F_tilde_s = np.reshape(ssps.FXw, (-1,)+ssps.par_dims)
        # print required size of the distribution funtion f,
        print(f'Distribution functions should have shape {ssps.par_dims}')
        # calculate area-element of ssp grid
        delta_z = ssps.delta_z
        delta_t = ssps.delta_t
        ssps.delta_zt = delta_z[:, np.newaxis] * delta_t[np.newaxis, :]
        self.ssps = ssps
        # get losvd and coeffients of Hermite polynomials
        losvd = losvds.GaussHermite()
        self.losvd = losvd
        self.max_order_hermite = max_order_hermite
        H_coeffs = losvd.get_hermite_polynomial_coeffients(max_order_hermite)
        self.H_coeffs = H_coeffs
        # set up frequency array for calculating FT of LOSVD
        nl = ssps.FXw.shape[0]
        omega = np.linspace(0, np.pi, nl) # LOSVD real valued -> 0 < omega < pi
        omega /= dv
        self.omega = omega

    def validate_input(self, f, Theta_v):
        # check that f is correct shape
        ssp_grid_shape = self.ssps.par_dims
        error_msg = f"f must have shape as SSP grid, i.e {ssp_grid_shape}"
        assert f.shape == ssp_grid_shape, error_msg
        # check that Theta_v has enough entries
        error_msg = "Theta_v must have > 2 entries"
        assert len(Theta_v) > 2, error_msg
        # check that sigma is positive
        sigma = Theta_v[1]
        error_msg = "sigma must be > 0"
        # check that h is correct length
        h = Theta_v[2:]
        error_msg = f"h must have length {self.max_order_hermite+1}"
        assert len(h)==self.max_order_hermite+1, error_msg

    def unpack_Theta_v(self, Theta_v):
        V = Theta_v[0]
        sigma = Theta_v[1]
        h = Theta_v[2:]
        M = len(h) - 1
        return V, sigma, h, M

    def evaluate(self, f, Theta_v):
        """Evaluate the observed (noise-free) signal

        Given a distribution funtion ``f` and parameters ``Theta_v`` of a Gauss
        Hermite LOSVD, evluate the observed signal. The signal is the
        convolution of the LOSVD with the composite spectrum. This convolution
        is calculated using Fourier transforms (FTs).

        Parameters
        ----------
        f
            2D array with dimensions compatible with the SSP grid
            The distribution function of (non-negative) weights of stellar-
            populations used to create the composite spectrum
        Theta_v : array-like
            Parameters of the Gauss Hermite LOSVD.

        Returns
        -------
        array_like
            The observed (noise-free) signal

        """
        self.validate_input(f, Theta_v)
        # get the FT of the composite spectrum
        # this is the integral of the FT of the templates (i.e. F_tilde_s)
        # weighted by the distribution funtion f
        F_tilde_S = np.sum(f * self.ssps.F_tilde_s * self.ssps.delta_zt, (1,2))
        # get the FT of the LOSVD
        # see method losvds.GaussHermite.evaluate_fourier_transform for details
        V, sigma, h, M = self.unpack_Theta_v(Theta_v)
        F_losvd = self.losvd.evaluate_fourier_transform(self.H_coeffs,
                                                        V,
                                                        sigma,
                                                        h,
                                                        M,
                                                        self.omega)
        # pointwise-product of the FTs gives the FT of the convolution
        F_ybar = F_tilde_S * F_losvd
        # IFT of F_ybar gives the signal ybar
        ybar = np.fft.irfft(F_ybar, self.ssps.n_fft)
        return ybar

    def partial_derivative_wrt_Theta_v(self, f, Theta_v):
        """Evaluate partial derivatives of the operator wrt Theta_v

        Parameters
        ----------
        f : array_like
            2D array with dimensions compatible with the SSP grid
            The distribution function of (non-negative) weights of stellar-
            populations used to create the composite spectrum
        Theta_v : array_like
            Parameters of the Gauss Hermite LOSVD

        Returns
        -------
        array ddTheta_V_ybar with shape (len(_y), len(Theta_v))
            where
            - ddTheta_V_ybar[:,0] is partial derivative wrt V
            - ddTheta_V_ybar[:,1] is partial derivative wrt sigma
            - ddTheta_V_ybar[:,2:] are partial derivatives wrt h_0,..., h_M
        """
        self.validate_input(f, Theta_v)
        # get the FT of the composite spectrum
        # this is the integral of the FT of the templates (i.e. F_tilde_s)
        # weighted by the distribution funtion f
        F_tilde_S = np.sum(f * self.ssps.F_tilde_s * self.ssps.delta_zt, (1,2))
        # get the partial derivatives of the FT of the LOSVD
        # see method losvds.GaussHermite.evaluate_fourier_transform for details
        V, sigma, h, M = self.unpack_Theta_v(Theta_v)
        ddTheta_V_F_losvd = self.losvd.partial_derivs_of_ft_wrt_Theta_v(
            self.H_coeffs,
            V,
            sigma,
            h,
            M,
            self.omega)
        # pointwise-product of the FTs gives the FT of the convolution
        ddTheta_V_F_ybar = (F_tilde_S * ddTheta_V_F_losvd.T).T
        # IFT of F_ybar gives the signal ybar
        ddTheta_V_ybar = np.fft.irfft(ddTheta_V_F_ybar, self.ssps.n_fft, axis=0)
        return ddTheta_V_ybar




# end
