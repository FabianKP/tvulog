"""
One inference-class to bind them all.
"""

from jax import random
import numpy as np
from typing import Literal, Optional

from examples.stellar_recovery.src.uq4pk_fit.regop import DiagonalOperator
from ..ornstein_uhlenbeck import OrnsteinUhlenbeck
from ..optimization import NonnegativeLinearModel
from ..svd_mcmc import SVD_MCMC
from ._forward_operator import ForwardOperator


class StatModel:
    """
    Abstract base class for that manages the optimization problem, regularization, and optionally also the
    uncertainty quantification.
    The full statistical model is::

        y ~ fwd(f) + error,
        error ~ normal(0, y_sd **2 * identity),
        f ~ normal(f_bar, cov1), where cov1 = (beta1 * P @ P.T)^(-1),
        f >= 0.

    Attributes
    -------
    beta
        The regularization parameter. Defaults to `1000 * snr`.
    P
        The regularization operator. Defaults to `OrnsteinUhlenbeck(m=self.m_f, n=self.n_f, h=h)`.

    """
    def __init__(self, y: np.array, y_sd: np.array, forward_operator: ForwardOperator):
        """
        Parameters
        ----------
        y : shape (n, )
            The masked data vector.
        y_sd : shape (n, )
            The masked vector of standard deviations
        forward_operator :
            The operator that maps stellar distribution functions to corresponding observations.
            Must satisfy `forward_operator.dim_y = n`.
        """
        # Check that the dimensions match.
        m = y.size
        assert y_sd.size == m
        assert forward_operator.dim_y == m
        self._op = forward_operator
        y_sum = np.sum(y)
        y_scaled = y / y_sum
        y_sd_scaled = y_sd / y_sum
        self._y = y_scaled
        self._scaling_factor = y_sum
        self._sigma_y = y_sd_scaled
        self._R = DiagonalOperator(dim=y.size, s=1 / y_sd_scaled)
        # get parameter dimensions from misfit handler
        self._m_f = forward_operator.m_f
        self._n_f = forward_operator.n_f
        self._dim_f = self._m_f * self._n_f
        self._dim_y = y.size
        self._snr = np.linalg.norm(y_scaled) / np.linalg.norm(y_sd_scaled)
        # SET DEFAULT PARAMETERS
        self._lb_f = np.zeros(self._dim_f)
        self._scale = self._dim_y
        # SET DEFAULT REGULARIZATION PARAMETERS
        self.beta = 1e3 * self._snr
        self.f_bar = np.zeros(self._dim_f) / self._scaling_factor
        h = np.array([4., 2.])
        self.P = OrnsteinUhlenbeck(m=self._m_f, n=self._n_f, h=h)

    @property
    def y(self) -> np.ndarray:
        """
        The RESCALED masked data vector.
        """
        return self._y

    @property
    def sigma_y(self) -> np.ndarray:
        """
        The RESCALED masked vector of standard deviations.
        """
        return self._sigma_y

    @property
    def m_f(self) -> int:
        """
        Number of rows for the image of a distribution function.
        """
        return self._m_f

    @property
    def n_f(self) -> int:
        """
        Number of columns for the image of a distribution function.
        """
        return self._n_f

    @property
    def dim_f(self) -> int:
        """
        Dimension (=no. of pixels) of the distribution function. This is just `m_f * n_f`.
        """
        return self._dim_f

    @property
    def dim_y(self) -> int:
        """
        Dimension of the masked data vector. This is just `y.size`.
        """
        return self._dim_y

    @property
    def snr(self) -> float:
        """
        The signal-to-noise ratio `||y||/||sigma_y||`.
        """
        return self._snr

    def compute_map(self) -> np.ndarray:
        """
        Computes the MAP estimator for the model.

        Returns
        ---
        f_map_image : shape (m_f, n_f)
            The MAP estimate (in image format).
        """
        model = NonnegativeLinearModel(y=self.y, P_error=self._R, G=self._op.mat, P_f=self.P, beta=self.beta,
                                       f_bar=self.f_bar, scaling_factor=self._scale)
        f_map = self._scaling_factor * model.fit()
        f_map_image = f_map.reshape(self.m_f, self.n_f)
        return f_map_image

    def negative_log_pdf(self, f: np.ndarray) -> float:
        """
        The negative logarithm of the (unconstrained) probability density function, modulo an additive constant::

            g(f) = ||(fwd(f) - y) / y_sd||^2 + beta * ||P f ||^2.

        Parameters
        ---
        f : shape (m_f n_f, ) or (m_f, n_f)
            The stellar distribution function. Can be provided in flattened or in image form.

        Returns
        ---
        log_prob
            The value `g(f)`.
        """
        fvec = f.flatten()
        misfit = np.sum(np.square(self._R.fwd(self._op.mat @ fvec - self.y)))
        prior = self.beta * np.sum(np.square(self.P.fwd(fvec - self.f_bar)))
        g = 0.5 * (misfit + prior)
        return g

    def sample_posterior(self, num_warmup: int, num_samples: int, method: Literal["svd", "full"], q: Optional[int] = 15,
                         rng_key: Optional[int] = None) -> np.ndarray:
        """
        Generates posterior samples using either SVD-MCMC or full HMC.

        Parameters
        ----------
        num_warmup
            Number of warmup steps in MCMC ("burn-in").
        num_samples
            Number of samples that should be generated after warmup.
        method :
            Method of MCMC sampling. "svd" for SVD-MCMC, "full" for HMC.
        q
            The latent dimension used in SVD-MCMC. Has no effect if `method="full"`.
        rng_key
            The key for the random number generation. Provide this to make output deterministic.

        Returns
        -------
        sample_array : shape (num_samples, m_f, n_f)
            The 3-dimensional array of samples. The first axis corresponds to the sample index, the other two to the
            image axes.
        """
        svd_mcmc_sampler = SVD_MCMC(G=self._op.mat, y=self.y, sigma_y=self.sigma_y, whiten=False)
        # Sample f.
        sigma_beta = self.P.cov / self.beta
        if method == "svd":
            assert q is not None
            # Choose degrees of freedom for reduced problem.
            svd_mcmc_sampler.set_q(q)
            print(f"Using SVD-MCMC with q={q}.")
            f_model = svd_mcmc_sampler.get_svd_reduced_model(Sigma_beta_tilde=sigma_beta)
        elif method == "full":
            print(f"Using full HMC.")
            f_model = svd_mcmc_sampler.get_full_model(Sigma_beta_tilde=sigma_beta)
        else:
            raise ValueError("Unknown 'method'.")
        f_sampler = svd_mcmc_sampler.get_mcmc_sampler(f_model, num_warmup=num_warmup, num_samples=num_samples)
        # Run sampler.
        if rng_key is None:
            rng_key = random.PRNGKey(np.random.randint(0, 999999))
        f_sampler.run(rng_key)
        f_sampler.print_summary()
        f_array = f_sampler.get_samples()["beta_tilde"]
        # Reshape array into image format.
        sample_array = self._scaling_factor * f_array.reshape(-1, self.m_f, self.n_f)
        # Return samples
        return sample_array
