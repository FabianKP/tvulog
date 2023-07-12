import numpy as np
from scipy import stats

class RandomGMM_DistributionFunction:

    def __init__(self,
                 modgrid=None,
                 fill_method='random_gblobs',
                 fill_pars={},
                 normalise=False):
        # make DF array
        self.shape = modgrid.par_dims
        self.dim = modgrid.npars
        self.par_mask_nd = modgrid.par_mask_nd
        self.par_idx = modgrid.par_idx
        self.F = np.zeros(self.shape)
        xarr = []
        for n_p in self.shape:
            if n_p>1:
                xarr += [np.arange(n_p)/(n_p-1.)]
            elif n_p==1:
                xarr += [np.array([0.5])]
        xarr = np.meshgrid(*xarr, indexing='ij')
        xarr = np.array(xarr)
        self.xarr = xarr
        # fill DF
        if fill_method=='constant':
            self.fill_constant(**fill_pars)
        elif fill_method=='uniform_random':
            self.fill_uniform_random(**fill_pars)
        elif fill_method=='gblob':
            self.fill_gblob(**fill_pars)
        elif fill_method=='random_gblob':
            self.fill_random_gblob(**fill_pars)
        elif fill_method=='random_gblobs':
            self.fill_random_gblobs(**fill_pars)
        else:
            raise ValueError('Unknown fill_method')
        # mask, normalise and flatten DF
        self.masker()
        self.normalise = normalise
        if normalise:
            self.normaliser()
        self.make_beta_vector()

    def masker(self):
        self.F[self.par_mask_nd] = 0.

    def normaliser(self):
        self.F /= np.sum(self.F)

    def make_beta_vector(self):
        idx = tuple([self.par_idx[i] for i in range(self.dim)])
        self.beta = self.F[idx]

    def fill_constant(self, k):
        self.F += k

    def fill_uniform_random(self, norm):
        self.F += np.random.uniform(0, norm, size=self.shape)

    def fill_gblob(self, norm, mu, cov):
        mvn = stats.multivariate_normal(mu, cov)
        pdf = mvn.pdf(self.xarr.reshape(self.dim, -1).T).T
        self.F += norm*pdf.reshape(self.shape)

    def fill_random_gblob(self,
                          lognorm_lo=0,
                          lognorm_hi=1.5,
                          slope_lognorm=10.,
                          mu_lo=0,
                          mu_hi=1,
                          logsig_lo=-2.,
                          logsig_hi=0.,
                          slope_logsig=10.):
        # get random gaussian paramters
        lognorm = np.random.uniform(lognorm_lo, lognorm_hi, size=1)
        norm = slope_lognorm**lognorm
        mu = np.random.uniform(mu_lo, mu_hi, self.dim)
        # compose covariance matrix from random eigenvalues and rotation
        logsim = np.random.uniform(logsig_lo, logsig_hi, size=self.dim)
        sigma = slope_logsig**logsim
        Sigma = np.diag(sigma**2.)
        if self.dim>1:
            R = stats.special_ortho_group.rvs(self.dim)
            cov = np.einsum('ij,jk,kl', R, Sigma, R.T)
        else:
            cov = sigma**2.
        self.fill_gblob(norm, mu, cov)

    def fill_random_gblobs(self,
                           n=3,
                           lognorm_lo=0,
                           lognorm_hi=1.5,
                           slope_lognorm=2,
                           mu_lo=0,
                           mu_hi=1,
                           logsig_lo=-2.,
                           logsig_hi=0.,
                           slope_logsig=5):
        for i in range(n):
            self.fill_random_gblob(lognorm_hi=lognorm_lo,
                                   slope_lognorm=slope_lognorm,
                                   mu_lo=mu_lo,
                                   mu_hi=mu_hi,
                                   logsig_lo=logsig_lo,
                                   logsig_hi=logsig_hi,
                                   slope_logsig=slope_logsig)
