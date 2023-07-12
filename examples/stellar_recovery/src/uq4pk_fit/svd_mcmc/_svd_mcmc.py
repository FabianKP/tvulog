"""
Courtesy of Prashin Jethwa
"""

import numpy as np
import os

NUM_CPU = 10         # Number of CPUs used for computations.
# Enforce parallel usage of CPU.
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CPU}"

import jax
import numpyro

# Get number of available CPUs.
numpyro.set_platform("cpu")
NUM_CHAINS = jax.local_device_count()
print(f"Using {NUM_CHAINS} CPUs for parallel sampling.")
from jax.lib import xla_bridge
print(f"JAX device: {xla_bridge.default_backend()}")

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
numpyro.set_platform("cpu")
from numpyro.infer import MCMC, NUTS


class SVD_MCMC:

    def __init__(self, G: np.ndarray, y: np.ndarray, sigma_y: np.ndarray, whiten: bool = True):
        """

        Parameters
        ----------
        G : shape (M, N)
            The matrix representation of the forward operator that maps stellar distribution functions to the (masked)
            light spectra.
        y : shape (M, )
            The observed (masked) spectrum.
        sigma_y : shape (M, )
            The vector of (masked) noise standard deviations.
        whiten
            If True, the forward operator is whitened for easier sampling.
        """
        self.X = G
        n, p = G.shape
        self._n = n
        self._p = p
        self._y = y
        self._sigma_y = sigma_y
        self._whiten = whiten
        self.whiten_X()
        self.do_svd()

    def whiten_X(self):
        if self._whiten:
            sum_y = np.sum(self._y)
            sum_x_j = np.sum(self.X, 0)
            X_tmp = sum_y * self.X / sum_x_j
            self.mu = np.mean(X_tmp, 1)
            self.X_tilde = (X_tmp.T - self.mu).T
            self.D = np.diag(sum_x_j / sum_y)
            self.Dinv = np.diag(sum_y / sum_x_j)
            self.X_lw = np.dot(self.X, self.Dinv)
        else:
            self.mu = np.zeros(self.X.shape[0])
            self.X_tilde = self.X
            self.D = np.identity(self.X.shape[1])
            self.Dinv = self.D
            self.X_lw = self.X

    def do_svd(self):
        U, Sig, VT = np.linalg.svd(self.X_tilde)
        self.U = U
        self.Sig = Sig
        self.VT = VT

    def set_q(self, q):
        self.q = q
        self.Z = np.dot(self.U[:,:q], np.diag(self.Sig[:q]))
        self.H = self.VT[:q,:]

    def get_mcmc_sampler(self, model, num_warmup=500, num_samples=500):
        kernel = NUTS(model)
        samples_per_chain = int(num_samples / NUM_CHAINS)
        mcmc_sampler = MCMC(kernel,
                            num_warmup=num_warmup,
                            num_samples=samples_per_chain,
                            num_chains=NUM_CHAINS)
        return mcmc_sampler

    def get_full_model(self, Sigma_beta_tilde=None):
        mu_beta_tilde = np.zeros(self._p)
        def full_model(Sigma_beta_tilde=Sigma_beta_tilde,
                                    y_obs=None):
            # hack for non-negativity
            beta_tilde = numpyro.sample("beta_tilde",
                                        dist.TruncatedNormal(0, 10., low=0),
                                        sample_shape=(self._p,))
            # prior
            beta_prior = dist.MultivariateNormal(mu_beta_tilde,
                                                 Sigma_beta_tilde)
            numpyro.factor("regulariser", beta_prior.log_prob(beta_tilde))
            # likelihood
            ybar = jnp.dot(self.X_lw, beta_tilde)
            nrm = dist.Normal(loc=ybar, scale=self._sigma_y)
            y_obs = numpyro.sample("y_obs", nrm, obs=self._y)
            return y_obs
        return full_model

    def get_svd_reduced_model(self, Sigma_beta_tilde=None):
        mu_beta_tilde = np.zeros(self._p)
        def svd_reduced_model(Sigma_beta_tilde=Sigma_beta_tilde,
                                    y_obs=None):
            # hack for non-negativity
            beta_tilde = numpyro.sample("beta_tilde",
                                        dist.TruncatedNormal(0., 10., low=0),
                                        sample_shape=(self._p,))
            # prior
            beta_prior = dist.MultivariateNormal(mu_beta_tilde,
                                                 Sigma_beta_tilde)
            numpyro.factor("regulariser", beta_prior.log_prob(beta_tilde))
            # likelihood
            eta = jnp.dot(self.H, beta_tilde)
            alpha = jnp.sum(beta_tilde)
            # likelihood
            ybar = alpha * self.mu + jnp.dot(self.Z, eta)
            nrm = dist.Normal(loc=ybar, scale=self._sigma_y)
            y_obs = numpyro.sample("y_obs", nrm, obs=self._y)
            return y_obs
        return svd_reduced_model
