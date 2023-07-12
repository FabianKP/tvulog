"""
Courtesy of Prashin Jethwa.
"""

import numpy as np
from scipy import stats

class Samples:
    """Class for samples. Contains methods to calculate:
    1) MAP, i.e. the the maximum probability sample
    2) average 1D marginalised samples, either:
        i) median
        ii) mode estimtaed by fitting beta distribution
        iii) mode estimtaed as the "halfsample mode"
    3) Bayesian Credible Intervals (BCIs) - i.e. intervals that contain/exclude
    a specified fraction of samples, either:
        i) even-tailed interval
        ii) highest-density interval
    4) gaussian_significances_1d
    Parameters
    ----------
    x : array of (n_smp, smp_shape)
        sample values
    logprob : array (n_smp,)
        log proabability values
    extras
        ???
    """
    def __init__(self, x: np.ndarray = None, logprob: np.ndarray = None, extras=None):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self.x = x
        self.n_smp = x.shape[0]
        self.smp_shape = x.shape[1::]
        self.smp_dim = len(self.smp_shape)
        self.sx = np.sort(x, axis=0)
        self.logprob = logprob
        self.extras = extras

    def check_p_in_p_out(self, p_in, p_out):
        """
        Check p_in/p_out parameters for BCIs i.e. the fraction of samples
        to include/exclude in the intervals.
        """
        if (p_in is None) + (p_out is None) != 1:
            raise ValueError('Exactly one of p_in or p_out must be set')
        if p_in is None:
            p_out = np.atleast_1d(p_out)
            p_in = 1. - p_out
        if p_out is None:
            p_in = np.atleast_1d(p_in)
            p_out = 1. - p_in
        return p_in, p_out

    def get_MAP(self):
        idx = np.argmax(self.logprob)
        idx = (idx,) + tuple([slice(0,i,1) for i in self.smp_shape])
        map = self.x[idx]
        return map

    def get_median(self):
        median = np.percentile(self.x, 50., axis=0)
        return median

    def get_percentile(self, p):
        perc = np.percentile(self.x, p, axis=0)
        return perc

    def get_betafit_mode(self):
        """
        Estimate mode of marginalised 1D distributions by fitting beta
        distribution.

        Returns
        -------
        arr mode_betafit, with shape = smp_shape
            contains NaN where distribution is estimated to be flat/multi-modal

        """
        if hasattr(self, 'betafit_mode'):
            return self.betafit_mode
        else:
            def beta_mode(a, b, loc, scale):
                a, b = np.atleast_1d(a),  np.atleast_1d(b)
                # mode for unit beta i.e. with support (0,1)
                mode = np.empty(a.shape)
                mode[:] = np.nan
                count = 0
                idx = np.where((a>1) & (b>1))
                if (idx[0].size>0) and (count < a.size):
                    mode[idx] = (a[idx]-1.)/(a[idx]+b[idx]-2)
                    count += idx[0].size
                idx = np.where((a<=1) & (b>1))
                if (idx[0].size>0) and (count < a.size):
                    mode[idx] = 0.
                    count += idx[0].size
                idx = np.where((a>1) & (b<=1))
                if (idx[0].size>0) and (count < a.size):
                    mode[idx] = 1.
                    count += idx[0].size
                if count<a.size:
                    print('Some samples have no mode i.e bimodal or uniform')
                # rescale and shift
                mode *= scale
                mode += loc
                return mode
            x = self.x.reshape((self.n_smp, -1))
            shape = (x.shape[1],)
            a = np.full(shape, np.nan)
            b = np.full(shape, np.nan)
            loc = np.full(shape, np.nan)
            scale = np.full(shape, np.nan)
            for i, x0 in enumerate(x.T):
                a[i], b[i], loc[i], scale[i] = stats.beta.fit(x0)
            mode = beta_mode(a, b, loc, scale)
            mode = mode.reshape(self.smp_shape)
            self.betafit_mode = mode
            return self.betafit_mode

    def get_halfsample_mode(self, axis=None, dtype=None):
        """
        Algorithm to estimate mode given 1D samples. Faster than betafit_mode.

        Returns
        -------
        arr halfsample_mode, with shape = smp_shape

        """
        # find nested highest-density bayesian credible intervals (HDBCI) till
        # only 1 or 2 samples remain
        n_int = self.n_smp  # number of samples inside HDBCI
        start = np.zeros(shape=self.smp_shape, dtype=int) # start index of HDBCI
        while n_int>2:  # ... continue till 1 or 2 samples remain
            # get "idx_int" i.e. indices of samples inside HDBCI
            # 1) get index array of right shape + relative index positions
            tmp = (np.arange(0,n_int),)
            tmp = tmp + tuple([np.arange(0,n) for n in self.smp_shape])
            idx_int = np.meshgrid(*tmp, indexing='ij')
            # 2) then offset by starting index of current HDBCI
            offset = np.broadcast_to(start, (n_int,)+self.smp_shape)
            idx_int[0] += offset
            idx_int = tuple(idx_int)
            # 3) get starting index of the next HDBCI
            tmp, n_int = self.highest_density_single_bci(self.sx[idx_int],
                                                         0.5,
                                                         return_idx_and_n=True)
             # offset previous starting position by new one
            start += tmp
        # return average of remaining samples
        if n_int==1:
            tmp = tuple([np.arange(0,n) for n in self.smp_shape])
            idx_mode = np.meshgrid(*tmp, indexing='ij')
            idx_mode = tuple([start] + idx_mode)
            mode = self.sx[idx_mode]
        elif n_int==2:
            tmp = tuple([np.arange(0,n) for n in self.smp_shape])
            idx_mode = np.meshgrid(*tmp, indexing='ij')
            idx_mode_0 = tuple([start] + idx_mode)
            idx_mode_1 = tuple([start+1] + idx_mode)
            mode = (self.sx[idx_mode_0] + self.sx[idx_mode_1])/2.
        else:
            raise ValueError('n_int =/= 1 or 2 - something has gone wrong')
        return mode

    def get_mad_from_mode(self, axis=None, dtype=None):
        '''
        median absolute deviation from the mode
        '''
        mode = self.get_halfsample_mode(axis=axis, dtype=dtype)
        abs_dev = np.abs(self.x - mode)
        mad = np.median(abs_dev, axis=0)
        return mad

    def get_mad_from_median(self, axis=None, dtype=None):
        '''
        median absolute deviation from the mode
        '''
        median = self.get_median()
        abs_dev = np.abs(self.x - median)
        mad = np.median(abs_dev, axis=0)
        return mad

    def even_tailed_bcis(self, p_in=None, p_out=None):
        """
        Get even-tailed BCIs for 1D marginalised distributions, i.e.
        fraction (1-p_out)/2. of samples either side of interval
        """
        p_in, p_out = self.check_p_in_p_out(p_in, p_out)
        p_lim_even_tailed = np.concatenate((p_out/2., 1.-p_out/2.))
        bcis = np.percentile(self.x, 100.*p_lim_even_tailed, axis=0)
        bcis = np.reshape(bcis, (2, len(p_in))+self.smp_shape)
        bcis = np.swapaxes(bcis, 0, 1)
        return bcis

    @staticmethod
    def highest_density_single_bci(sx, p_in,return_idx_and_n=False):
        """
        Get highest-density BCIs for 1D marginalised distributions, i.e. the
        smallest continuous interval a containing fraction p_in of samples.
        Adapted from PYMC3.

        Parameters
        ----------
        sx
            sorted samples
        p_in
            fraction inside desired BCI
        return_idx_and_n
            return option
        """
        n_smp = sx.shape[0]
        smp_shape = sx.shape[1::]
        n_smp_in = int(np.floor(p_in * n_smp))
        n_smp_out = n_smp - n_smp_in
        possible_widths = sx[n_smp_in:] - sx[:n_smp_out]
        if len(possible_widths) == 0:
            raise ValueError('Too few elements for interval calculation')
        # argmin for minimum widths of first axis, i.e. sample axis
        min_width_idx = np.argmin(possible_widths, 0)
        min_width_idx = np.ravel(min_width_idx)
        # construct tuple for indices of remaining axes of sx
        idx_x_shape = [np.arange(i) for i in smp_shape]
        idx_x_shape = np.meshgrid(*idx_x_shape, indexing='ij')
        idx_x_shape = [np.ravel(idx) for idx in idx_x_shape]
        idx_x_shape = tuple(idx_x_shape)
        # get indices/values of bci edges
        idx_min = (min_width_idx,) + idx_x_shape
        idx_max = (min_width_idx+n_smp_in,) + idx_x_shape
        if return_idx_and_n:
            return np.reshape(idx_min[0], smp_shape), n_smp_in
        hdi_min = sx[idx_min]
        hdi_max = sx[idx_max]
        bci = np.array([hdi_min, hdi_max])
        return np.reshape(bci, (2,)+smp_shape)

    def highest_density_bcis(self,
                             p_in=None,
                             p_out=None):
        """
        Loop over highest_density_single_bci for different p_in
        """
        p_in, p_out = self.check_p_in_p_out(p_in, p_out)
        bcis = np.zeros((len(p_in), 2) + self.smp_shape)
        idx = tuple([slice(0,end) for end in self.smp_shape])
        for i, p_in0 in enumerate(p_in):
            bcis[(i,)+idx] = self.highest_density_single_bci(self.sx, p_in0)
        return bcis

    def gaussian_significances_1d(self,
                                  truth=None,
                                  center=None,
                                  return_center=False,
                                  resid_nsmp_min=30):
        """
        Find gaussian significances of 1D residuals between truth and center

        Parameters
        ----------
        truth : array with shape = self.smp_shape
            whether to plot the true distribution function
        center : array with shape = self.smp_shape, or string
            central values of posterior, either precomputed array or one of:
            i) 'MAP'
            ii) 'median'
            iii) 'mode'
        return_center : boolean
            whether to return center
        resid_nsmp_min : int
            minimum number of samples to consider when calculting significances
            i.e. significance values capped at
            stats.halfnorm.ppf(1-resid_nsmp_min/nsmp_total)
        """
        # get center
        if isinstance(center, (list, tuple, np.ndarray)):
            center = np.array(center)
            if center.shape != self.smp_shape:
                raise ValueError('center is wrong shape')
        elif center=='MAP':
            center = self.get_MAP()
        elif center=='median':
            center = self.get_median()
        elif center=='mode':
            center = self.get_halfsample_mode()
        else:
            raise ValueError('Unknown option for center')
        # check truth
        if truth.shape != self.smp_shape:
            raise ValueError('truth shape wrong shape')
        diff = center - truth
        # make copies of samples, centers, truths with signs flipped where
        # appropritate such that all residuals are positive
        cent_pos = 1.*center
        true_pos = 1.*truth
        samp_pos = 1.*self.sx
        idx = np.where(diff < 0)
        if idx[0].size>0:
            cent_pos[idx] = -1.*center[idx]
            true_pos[idx] = -1.*truth[idx]
            for ids in zip(*idx):
                tmp = tuple([slice(0,self.n_smp)]+[id0 for id0 in ids])
                samp_pos[tmp] = -1.*self.sx[tmp]
        # count total number of samples below (positive flipped) center
        tot = (samp_pos<=cent_pos)
        tot = np.sum(tot, 0)
        # count number of samples between (positive flipped) truths and centers
        cnt = ((samp_pos>=true_pos) & (samp_pos<=cent_pos))
        cnt = np.sum(cnt, 0)

        def my_halfnorm_cdf(ncnt, ntot, nmin):
            """
            CDF for halformal distribution corrected for small numbers,
            i.e. normalised such that maximum significance = stats.halfnorm.ppf(1-resid_nsmp_min/nsmp_total).
            """
            return (ncnt - nmin*np.exp((ncnt-ntot)/nmin-1.))/ntot

        halfnorm_cdf = my_halfnorm_cdf(cnt, tot, resid_nsmp_min)
        significance = stats.halfnorm.ppf(halfnorm_cdf)
        significance *= np.sign(diff)
        return significance
