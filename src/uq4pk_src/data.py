import numpy as np
from scipy import stats
from . import distribution_function, losvds
from astropy.io import fits
import spectres

class Noise:

    def __init__(self,
                 n,
                 sig=None,
                 sig_i=None,
                 cov=None):
        # check input
        self.n = n
        c1 = sig is not None
        c2 = sig_i is not None
        c3 = cov is not None
        if c1 + c2 + c3 != 1:
            errmsg = 'Exactly one of sig, sig_i or cov must be set'
            raise ValueError(errmsg)
        if c1:
            self.sig = sig
            self.sig_i = np.ones(self.n)*sig
            self.cov = sig**2 * np.identity(n)
            self.pre = sig**-2 * np.identity(n)
        if c2:
            assert sig_i.shape == (n,)
            self.sig_i = sig_i
            self.cov = np.diag(sig_i**2)
            self.pre = np.diag(sig_i**-2)
        if c3:
            assert cov.shape == (n,n)
            self.cov = cov
            self.pre = np.linalg.inv(cov)

    def sample(self):
        if hasattr(self, 'sig_i'):
            nrm = stats.norm()
            noise = nrm.rvs(size=self.n)*self.sig_i
        else:
            mvn = stats.multivariate_normal(mean=np.zeros(self.n),
                                            cov=self.cov)
            noise = mvn.rvs()
        return noise

    def transform_subscripts(self, subscripts):
        # split subscripts
        if '->' in subscripts:
            ss_in, ss_out = subscripts.split('->')
            has_output = True
        else:
            ss_in = subscripts
            has_output = False
        ss_x, ss_cov, ss_y = subscripts.split(',')

        # transform subscripts
        if hasattr(self, 'sig'):
            ss_x = ss_x.replace(ss_cov[0], ss_cov[1])
            ss = f'{ss_x},{ss_y}'
        # isotropic case
        elif hasattr(self, 'sig_i'):
            ss_x = ss_x.replace(ss_cov[0], ss_cov[1])
            ss = f'{ss_x},{ss_cov[1]},{ss_y}'
        # general case
        else:
            ss = ss_in
        if has_output is True:
            ss + '->' + ss_out
        return ss

    def einsum_x_cov_y(self, subscripts, x, y, precision=False):
        '''
        Fast evaluation of products with covariance or precision matrix.
        Optimised for cases of istotropic/diagonal/general noise.
        Evaluates
            np.einsum(subscripts, x, cov, _y) if precision=False
            np.einsum(subscripts, x, pre, _y) if precision=True
        Subscripts must be a valid einsum string of the type
            '{sx}i,ij,j{sy}'
        possibly appended with a valid output string of the type
            '->{s_out}'
        '''
        ss = self.transform_subscripts(subscripts)
        if hasattr(self, 'sig'):
            if precision:
                xcy = self.sig**-2. * np.einsum(ss, x, y, optimize='true')
            else:
                xcy = self.sig**2. * np.einsum(ss, x, y, optimize='true')
        # isotropic case
        elif hasattr(self, 'sig_i'):
            if precision:
                xcy = np.einsum(ss, x, self.sig_i**-2., y, optimize='true')
            else:
                xcy = np.einsum(ss, x, self.sig_i**2., y, optimize='true')
        # general case
        else:
            if precision:
                xcy = np.einsum(ss, x, self.pre, y, optimize='true')
            else:
                xcy = np.einsum(ss, x, self.cov, y, optimize='true')
        return xcy


class Data:

    def __init__(self,
                 lmd=None,
                 y=None):
        n = lmd.size
        if lmd.size!=y.size:
            errmsg = 'lmd and _y must have same length'
            raise ValueError(errmsg)
        self.n = n
        self.lmd = lmd
        self.y = y


class MockData(Data):

    def __init__(self,
                 ssps=None,
                 df=None,
                 losvd=None,
                 snr=None):
        n, p = ssps.X.shape
        self.ssps = ssps
        self.df = df
        self.S = np.dot(ssps.X, df.beta)
        self.losvd = losvd
        self.ybar = losvd.convolve(S=self.S, lmd_in=ssps.lmd)
        self.snr = snr
        sig = np.mean(np.abs(self.ybar))/snr
        noise = Noise(n, sig=sig)
        y = self.ybar + noise.sample()
        super().__init__(lmd=ssps.lmd,
                         y=y)


class RandomMockData(MockData):

    def __init__(self,
                 ssps=None,
                 snr=None):
        # generate random DF and losvd
        df = distribution_function.DistributionFunction(ssps)
        losvd = losvds.InputLOSVD()
        super().__init__(ssps=ssps,
                         df=df,
                         losvd=losvd,
                         snr=snr)


class MockData(Data):

    def __init__(self,
                 ssps=None,
                 df=None,
                 losvd=None,
                 snr=None):
        n, p = ssps.X.shape
        self.ssps = ssps
        self.df = df
        self.S = np.dot(ssps.X, df.beta)
        self.losvd = losvd
        self.ybar = losvd.convolve(S=self.S, lmd_in=ssps.lmd)
        self.snr = snr
        sig = np.mean(np.abs(self.ybar))/snr
        noise = Noise(n, sig=sig)
        y = self.ybar + noise.sample()
        super().__init__(lmd=ssps.lmd,
                         y=y)


class M54(Data):

    def __init__(self):
        self.datadir = losvds.__file__.replace('losvds.py', '../data/M54/')
        self.read_spectrum()
        self.read_ppxf_map_solution()
        self.read_mcsims_ppxf_map_solution()
        self.read_ground_truth()

    def read_spectrum(self):
        # observed spectrum of M54 (from a collapsed MUSE cube)
        fname = self.datadir + 'M54_integrated_spectrum_from_stars_member_new2.fits'
        hdu = fits.open(fname)
        lmd = hdu[0].data
        spectrum = hdu[1].data
        noise = hdu[2].data
        # truncate max wavelength
        idx = np.where(lmd<=8951)
        lmd = lmd[idx]
        spectrum = spectrum[idx]
        noise = noise[idx]
        # mask bad pixels
        mask = np.ones_like(lmd, dtype=bool)
        mask[(lmd>=5850) & (lmd<=5950)] = False
        mask[(lmd>=6858.7) & (lmd<=6964.9)] = False
        mask[(lmd>=7562.3) & (lmd<=7695.6)] = False
        # store
        super().__init__(lmd=lmd, y=spectrum)
        self.noise_level = noise
        self.mask = mask

    def read_ppxf_map_solution(self):
        # The npz files contain the weights returned by ppxf, stored under the keyword „regul“. E.g. I do:
        fname = self.datadir + 'M54_integrated_spectrum_from_stars_member_new2_integrated_D1_EMILESBASTI_10_cut.npz'
        data = np.load(fname)
        ppxf_map_solution = data['regul'].reshape(53,12)
        self.ppxf_map_solution = ppxf_map_solution

    def read_mcsims_ppxf_map_solution(self):
        fname = self.datadir + 'M54_integrated_spectrum_from_stars_member_new2_integrated_D1_EMILESBASTI_cut_MC.npz'
        data = np.load(fname)
        mcsims_map_weights = data['regul'].reshape(53,12,-1)
        self.mcsims_map_weights = mcsims_map_weights

    def read_ground_truth(self):
        # The last file is Mayte’s data binned to the age and metallicity grid of the MILES models.
        fname = self.datadir + 'M54_single_stars_binned_to_miles.npz'
        data = np.load(fname)
        ground_truth = data['weights']
        self.ground_truth = ground_truth

    def logarithmically_resample(self, dv=30.):
        speed_of_light = 299792.
        dw = dv/speed_of_light
        w_in = np.log(self.lmd)
        w = np.arange(np.min(w_in), np.max(w_in)+dw, dw)
        lmd_new = np.exp(w)
        y, noise_level = spectres.spectres(lmd_new,
                                           self.lmd,
                                           self.y,
                                           self.noise_level)
        self.speed_of_light = speed_of_light
        self.dv = dv
        self.dw = dw
        self.w = w
        self.y = y
        self.lmd = lmd_new
        self.noise_level = noise_level
        # mask bad pixels
        mask = np.ones_like(lmd_new, dtype=bool)
        mask[(lmd_new>=5850) & (lmd_new<=5950)] = False
        mask[(lmd_new>=6858.7) & (lmd_new<=6964.9)] = False
        mask[(lmd_new>=7562.3) & (lmd_new<=7695.6)] = False
        self.mask = mask


# end
