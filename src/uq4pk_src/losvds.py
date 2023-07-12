import numpy as np
from scipy import stats, special, interpolate, integrate

class LOSVD(object):

    def __init__(self, convolve_method='fft'):
        self.speed_of_light = 299792. # in km/s
        self.enumerate_convolve_methods()
        self.set_convolve_method(convolve_method)

    def evaluate(self, v):
        """ dummy method to evaluate the LOSVD
        Parameters
        ----------
        v : array
            velocity in km/s
        """
        pass

    def enumerate_convolve_methods(self):
        methods = ['convolve_integrate_direct',
                   'convolve_integrate_transform',
                   'convolve_fft']
        if hasattr(self, 'analytic_fourier_transform'):
            methods += ['convolve_fft_analytic']
        self.available_convolve_methods = methods

    def set_convolve_method(self, method):
        if 'convolve_'+method in self.available_convolve_methods:
            pass
        else:
            raise ValueError('Unknown convolve method. Defaulting to FFT')
            method = 'fft'
        self.do_convolve = getattr(self, 'convolve_'+method)

    def convolve(self,
                 S=None,
                 lmd_in=None,
                 lmd_out=None,
                 v_lim=(-1000.,1000.),
                 **kwargs):
        """Convolve a spectrum with this LOSVD. This is a dummy method; specific
        implementations are given below. Choice of which method to use is set by
        'set_convolve_method' or during initialisation.

        Parameters
        ----------
        S : array
            the input spectrum
        lmd_in : array
            wavelengths [angstrom] for input spectrum S
        lmd_out : array
            wavelengths [angstrom] to evaluate convolution, default = lmd_in
        v_lim : array size two
            (v_min, v_max) [km/s] limits for benchmarks
        kwargs : dict
            other kwargs passed to specific convolve_methods

        Returns
        -------
        array
            the convolved spectrum evaluated at lmd_out
        """
        assert len(S)==len(lmd_in)
        if lmd_out is None:
            lmd_out = lmd_in
        v_lim = np.array(v_lim)
        args = S, lmd_in, lmd_out, v_lim
        result = self.do_convolve(*args, **kwargs)
        return result

    def convolve_integrate_direct(self, S, lmd_in, lmd_out, v_lim):
        """Numerically evaluate the convultuon, i.e. calculate
            int_{v_lim[0]}^{v_lim[1]} (1+v/c)^-1 S(lmd (1+v/c)^-1) LOSVD(v) dv
        looping over desired output values lmd_out
        """
        def Sinterp(lmd):
            f = interpolate.interp1d(lmd_in,
                                     S,
                                     kind='cubic',
                                     bounds_error=False,
                                     fill_value=0.)
            return f(lmd)
        def ybar(lmd):
            def integrand(v):
                scale = 1. + v/self.speed_of_light
                return Sinterp(lmd/scale) * self.evaluate(v) / scale
            v_min, v_max = v_lim
            result, _ = integrate.quad(integrand, v_min, v_max)
            return result
        convolved_spectrum = [ybar(lmd0) for lmd0 in lmd_out]
        convolved_spectrum = np.array(convolved_spectrum)
        return convolved_spectrum

    def convolve_integrate_transform(self, S, lmd_in, lmd_out, v_lim):
        """Tansforming variables [eqns (2-7) of Ocvirk et al 2005/STECKMAP]
            w = ln(lmd) and u = ln(1+v/c)
        and defining
            S_tilde(w) = S(exp(w)) and LOSVD_tilde(u) = LOSVD(c*(exp(u)-1))
        the integral in "convolve_integrate_direct" becomes
            c int_{u_lim[0]}^{u_lim[1]} S_tilde(w-u) LOSVD_tilde(u) du
            = c S_tilde * LOSVD_tilde
        """
        def S_tilde(w):
            lmd = np.exp(w)
            f = interpolate.interp1d(lmd_in,
                                     S,
                                     kind='cubic',
                                     bounds_error=False,
                                     fill_value=0.)
            return f(lmd)
        def LOSVD_tilde(u):
            v = self.speed_of_light * (np.exp(u) - 1.)
            return self.evaluate(v)
        def u(v):
            return np.log(1+v/self.speed_of_light)
        def ybar(w):
            def integrand(u):
                return S_tilde(w-u) * LOSVD_tilde(u)
            u_min, u_max = u(v_lim)
            result, _ = integrate.quad(integrand, u_min, u_max)
            result *= self.speed_of_light
            return result
        convolved_spectrum = [ybar(w0) for w0 in np.log(lmd_out)]
        convolved_spectrum = np.array(convolved_spectrum)
        return convolved_spectrum

    def convolve_fft(self, S, lmd_in, lmd_out, v_lim, n_smp_losvd=151):
        """Convolve using Fourier transforms. Starting from final equation given
        in docblock for "convolve_integrate_transform", i.e.
            ybar(w) = c F^{-1} [F(S_tilde(w)) . F(LOSVD_tilde(w))]
        where F is the Fourier transform and . is pointwise multiplication
        """
        # sample LOSVD
        # use odd n_smp so that convolution doesn't shift pixels by 1/2
        if n_smp_losvd % 2 == 0:
            n_smp_losvd += 1
        v = np.linspace(*v_lim, n_smp_losvd)
        losvd = self.evaluate(v)
        # logarithmically re-sample the spectrum using losvd velocity spacing
        dv = v[1]-v[0]
        dw = dv/self.speed_of_light
        w_in = np.log(lmd_in)
        f = interpolate.interp1d(w_in,
                                 S,
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=0.)
        w = np.arange(np.min(w_in), np.max(w_in)+dw, dw)
        S = f(w)
        # FFT convolution with padding
        n_pad = len(S) + n_smp_losvd - 1
        n_pad = 2**int(np.ceil(np.log2(n_pad)))
        FS, FL = np.fft.rfft(S, n_pad), np.fft.rfft(losvd, n_pad)
        Fybar = FS * FL
        ybar = np.fft.irfft(Fybar, n_pad)
        ybar *= dw * self.speed_of_light
        # get padded w array for output
        p = int((n_smp_losvd - 1)/2)
        w_pad = 1.*np.arange(-p, n_pad-p, 1)
        w_pad *= dw
        w_pad += np.min(w)
        # sample output at desired location
        f = interpolate.interp1d(w_pad,
                                 ybar,
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=0.)
        w_out = np.log(lmd_out)
        ybar = f(w_out)
        return ybar

    def convolve_fft_analytic(self, S, lmd_in, lmd_out, v_lim, n_smp_losvd=101):
        # TODO: evaluate FT analytically if LOSVD permits
        pass


class GMM1D(LOSVD):
    """Class to hold 1D Gaussian mixture models

    Parameters
    ----------
    weights : array (n_cmp,)
        componenent weights
    means : array (n_cmp,)
        componenent means
    sigmas : array (n_cmp,)
        componenent sigmas

    """
    def __init__(self,
                 weights=None,
                 means=None,
                 sigmas=None):
        assert (len(weights)==len(means)) & (len(means)==len(sigmas))
        self.n_cmp = len(weights)
        self.nrm = stats.norm(means, sigmas)
        self.weights = weights
        LOSVD.__init__(self)

    def evaluate(self, x):
        x = np.atleast_1d(x)
        y = self.weights * self.nrm.pdf(x[:,np.newaxis])
        y = np.sum(y, 1)
        return np.squeeze(y)


class RandomGMM1D(GMM1D):

    def __init__(self,
                 n_cmp=5,
                 log_sig_rng=(-0.5,1),
                 max_abs_mu=1.,
                 dirichlet_alpha=1.):
        weights = stats.dirichlet(dirichlet_alpha*np.ones(n_cmp)).rvs()[0,:]
        means = stats.uniform(-max_abs_mu, 2*max_abs_mu).rvs(size=n_cmp)
        log_sigmas = stats.uniform(*log_sig_rng).rvs(size=n_cmp)
        sigmas = 10.**log_sigmas
        super(RandomGMM1D, self).__init__(weights=weights,
                                          means=means,
                                          sigmas=sigmas)


class InputLOSVD(RandomGMM1D, LOSVD):

    def __init__(self,
                 n_cmp=5,
                 log_sig_rng=(-1,1),
                 max_abs_mu=1,
                 alpha=1,
                 vel_scale_rng=(50,300)):
        weights = stats.dirichlet(alpha*np.ones(n_cmp)).rvs()[0,:]
        means = stats.uniform(-max_abs_mu, 2*max_abs_mu).rvs(size=n_cmp)
        log_sigmas = stats.uniform(*log_sig_rng).rvs(size=n_cmp)
        sigmas = 10.**log_sigmas
        super(RandomGMM1D, self).__init__(weights=weights,
                                          means=means,
                                          sigmas=sigmas)
        Delta_vel_scale = vel_scale_rng[1] - vel_scale_rng[0]
        tmp = [vel_scale_rng[0], Delta_vel_scale]
        self.vel_scale = stats.uniform(*tmp).rvs()
        LOSVD.__init__(self)

    def evaluate(self, v):
        x = v/self.vel_scale
        pdf = super(RandomGMM1D, self).evaluate(x)/self.vel_scale
        return pdf


class Histogram(object):
    """Class to hold histograms

    Parameters
    ----------
    xedg : array (n_bins+1,)
        histogram bin edges
    y : array hist_shape+(n_bins,)
        histogram values
    normalise : bool, default=True
        whether to normalise to pdf

    Attributes
    ----------
    x : array (n_bins,)
        bin centers
    dx : array (n_bins,)
        bin widths
    normalised : bool
        whether or not has been normalised to pdf

    """
    def __init__(self, xedg, y, normalise=True):
        self.xedg = xedg
        self.x = (xedg[:-1] + xedg[1:])/2.
        self.dx = xedg[1:] - xedg[:-1]
        self.y = y
        if normalise:
            self.normalise()
        else:
            self.normalised = False

    def normalise(self):
        norm = np.sum(self.y * self.dx, axis=-1)
        self.norm = norm
        self.y = (self.y.T/norm.T).T
        self.normalised = True

    def evaluate(self, x_eval):
        idx = np.digitize(x_eval, self.xedg)
        return self.y[idx]


class GaussHermite(LOSVD):

    def __init__(self):
        pass

    def get_hermite_polynomial_coeffients(self, max_order=None):
        """Get coeffients for hermite polynomials normalised as in eqn 14 of
        Capellari 2016

        Parameters
        ----------
        max_order : int
            maximum order hermite polynomial desired
            e.g. max_order = 1 --> use h0, h1
            i.e. number of hermite polys = max_order + 1

        Returns
        -------
        array (max_order+1, max_order+1)
            coeffients[i,j] = coef of x^j in polynomial of order i

        """
        if max_order is None:
            max_order = self.n_gh
        coeffients = []
        for i in range(0, max_order+1):
            # physicists hermite polynomials
            coef_i = special.hermite(i)
            coef_i = coef_i.coefficients
            # reverse poly1d array so that j'th entry is coeefficient of x^j
            coef_i = coef_i[::-1]
            # scale according to eqn 14 of Capellari 16
            coef_i *= (special.factorial(i) * 2**i)**-0.5
            # fill poly(i) with zeros for 0*x^j for j>i
            coef_i = np.concatenate((coef_i, np.zeros(max_order-i)))
            coeffients += [coef_i]
        coeffients = np.vstack(coeffients)
        return coeffients

    def standardise_velocities(self, v, v_mu, v_sig):
        """

        Parameters
        ----------
        v : array
            input velocity array
        v_mu : array (n_regions,)
            gauss hermite v parameters
        v_sig : array (n_regions,)
            gauss hermite sigma parameters

        Returns
        -------
        array (n_regions,) + v.shape
            velocities whitened by array v_mu, v_sigma

        """
        v = np.atleast_2d(v)
        v_mu = np.atleast_1d(v_mu)
        v_sig = np.atleast_1d(v_sig)
        assert v_mu.shape==v_mu.shape
        w = (v.T - v_mu)/v_sig
        w = w.T
        return w

    def evaluate_hermite_polynomials(self,
                                     coeffients,
                                     w,
                                     standardised=True,
                                     v_mu=None,
                                     v_sig=None):
        """

        Parameters
        ----------
        coeffients : array (n_herm, n_herm)
            coefficients of hermite polynomials as given by method
            get_hermite_polynomial_coeffients
        w : array
            if standardised==True
                shape (n_regions, n_vbins), standardised velocities
            else
                shape (n_vbins,), physical velocities
                and arrays v_mu and v_sig with shape (n_regions,) must be set

        Returns
        -------
        array shape (n_hists, n_regions, n_vbins)
            Hermite polynomials evaluated at w in array of

        """
        if not standardised:
            w = self.standardise_velocities(w, v_mu, v_sig)
        result = np.polynomial.polynomial.polyval(w, coeffients.T)
        return result

    def evaluate(self, v, v_mu, v_sig, h):
        """

        Parameters
        ----------
        v : array
            input velocity array
        v_mu : array (n_regions,)
            gauss hermite v parameters
        v_sig : array (n_regions,)
            gauss hermite sigma parameters
        h : array (n_hists, n_regions, n_herm)
            gauss hermite expansion coefficients

        Returns
        -------
        array shape same as v
            values of gauss hermite expansion evaluated at v

        """
        # check input
        v_mu = np.atleast_1d(v_mu)
        v_sig = np.atleast_1d(v_sig)
        if h.ndim == 1:
            h = np.atleast_3d(h).reshape(1,1,-1)
        # evaluate
        w = self.standardise_velocities(v, v_mu, v_sig)
        n_herm = h.shape[2]
        max_order = n_herm - 1
        coef = self.get_hermite_polynomial_coeffients(max_order=max_order)
        nrm = stats.norm()
        hpolys = self.evaluate_hermite_polynomials(coef, w)
        losvd = np.einsum('ij,kil,lij->kij',
                          nrm.pdf(w),
                          h,
                          hpolys,
                          optimize=True)
        losvd = np.squeeze(losvd)
        return losvd

    def get_gh_expansion_coefficients(self,
                                      v_mu=None,
                                      v_sig=None,
                                      vel_hist=None,
                                      max_order=4):
        """Calcuate coeffients of gauss hermite expansion of histogrammed LOSVD
        around a given v_mu and v_sig i.e. evaluate qn 7 of vd Marel & Franx 93

        Parameters
        ----------
        v_mu : array (n_regions,)
            gauss hermite v parameters
        v_sig : array (n_regions,)
            gauss hermite sigma parameters
        vel_hist : Histogram object
            velocity histograms of orbits
            where vel_hist._y has shape (n_orbs, n_regions, n_vbins)
        max_order : int
            maximum order hermite polynomial desired in the expansion
            e.g. max_order = 1 --> use h0, h1
            i.e. number of hermite polys = max_order + 1

        Returns
        -------
        h : array (n_hists, n_regions, max_order+1)
            where h[i,j,k] is order k GH coeffient of histogram i in region j

        """
        # check input
        v_mu = np.atleast_1d(v_mu)
        v_sig = np.atleast_1d(v_sig)
        if vel_hist._y.ndim == 1:
            y = np.atleast_3d(vel_hist._y).reshape(1, 1, -1)
            vel_hist = Histogram(xedg=vel_hist.xedg, y=y)
        assert max_order>=0
        assert v_mu.shape[0]==vel_hist.y.shape[1]
        assert v_sig.shape[0]==vel_hist.y.shape[1]
        # calculate
        w = self.standardise_velocities(vel_hist.x, v_mu, v_sig)
        coef = self.get_hermite_polynomial_coeffients(max_order=max_order)
        nrm = stats.norm()
        hpolys = self.evaluate_hermite_polynomials(coef, w)
        h = np.einsum('ijk,jk,ljk,k->ijl',           # integral in eqn 7
                      np.atleast_2d(vel_hist.y),
                      nrm.pdf(w),
                      hpolys,
                      vel_hist.dx,
                      optimize=True)
        h *= 2 * np.pi**0.5                         # pre-factor in eqn 7
        losvd_unnorm = self.evaluate(vel_hist.x, v_mu, v_sig, h)
        gamma = np.sum(losvd_unnorm * vel_hist.dx, -1)
        h = (h.T/gamma.T).T
        h = np.squeeze(h)
        return h

    def evaluate_fourier_transform(self, coeffients, V, sigma, h, M, omega):
        """Evaluate analytic Fourier transfrom of Gauss Hermite LOSVD at
        frequencies omega

        Parameters
        ----------
        coeffients : array (n_herm, n_herm)
            coefficients of hermite polynomials as given by method
            get_hermite_polynomial_coeffients
        V : float
            parameter V of the GH-LOSVD
        sigma : float
            parameter sigma of the GH-LOSVD
        h : array-like
            coeffecients of the GH expansion, h[0] = h_0, etc...
        M : int
            maximum order H polynomial, i.e. M=4 => use h0,h1,h2,h3,h4
        omega : array-like
            frequencies where to evaluate the fourier transform of the GH-LOSVD

        Returns
        -------
        array-like
            The fourier transform of the LOSVD evaluated at omega

        """
        error_msg = f'h should have length {M+1}'
        assert len(h)==M+1, error_msg
        sigma_omega = sigma*omega
        exponent = -1j*omega*V - 0.5*(sigma_omega)**2
        F_gaussian_losvd = np.exp(exponent)
        H_m = np.polynomial.polynomial.polyval(sigma_omega, coeffients.T)
        i_to_the_m = np.full(M+1, 1j)**np.arange(M+1)
        F_gh_poly = np.sum(i_to_the_m * h * H_m.T, 1)
        F_losvd = F_gaussian_losvd * F_gh_poly
        return F_losvd

    def partial_derivs_of_ft_wrt_Theta_v(self,
                                         coeffients,
                                         V,
                                         sigma,
                                         h,
                                         M,
                                         omega):
        """Evaluate partial derivatives of FT of the losvd at frequencies omega

        Parameters
        ----------
        coeffients : array (n_herm, n_herm)
            coefficients of hermite polynomials as given by method
            get_hermite_polynomial_coeffients
        V : float
            parameter V of the GH-LOSVD
        sigma : float
            parameter sigma of the GH-LOSVD
        h : array-like
            coeffecients of the GH expansion, h[0] = h_0, etc...
        M : int
            maximum order H polynomial, i.e. M=4 => use h0,h1,h2,h3,h4
        omega : array-like
            frequencies where to evaluate the fourier transform of the GH-LOSVD

        Returns
        -------
        array-like
            The fourier transform of the LOSVD evaluated at omega

        """
        error_msg = f'h should have length {M+1}'
        assert len(h)==M+1, error_msg
        # get partial wrt h_m
        sigma_omega = sigma*omega
        exponent = -1j*omega*V - 0.5*(sigma_omega)**2
        F_gaussian_losvd = np.exp(exponent)
        H_m = np.polynomial.polynomial.polyval(sigma_omega, coeffients.T)
        i_to_the_m = np.full(M+1, 1j)**np.arange(M+1)
        wrt_hm = (F_gaussian_losvd * (i_to_the_m * H_m.T).T).T
        # get partial wrt V
        F_losvd = self.evaluate_fourier_transform(coeffients,V,sigma,h,M,omega)
        wrt_V = - 1j * omega * F_losvd
        # get partial wrt sigma
        tmp1 = - sigma * omega**2. * F_losvd
        m = np.arange(0, M+1)
        sqrt2m = (2.*m)**0.5
        tmp2 = np.sum(i_to_the_m[1:] * sqrt2m[1:] * h[1:] * H_m[:-1,:].T, 1)
        wrt_sigma = tmp1 + omega * F_gaussian_losvd * tmp2
        F_partial_derivatives = np.vstack((wrt_V, wrt_sigma, wrt_hm.T)).T
        return F_partial_derivatives

# end
