
import cuqi
from matplotlib import pyplot as plt
plt.style.use("examples/tv_ulog.mplstyle")
import matplotlib
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
from matplotlib import colors
import numpy as np
from pathlib import Path

from src.ulog.scale_space_tube import estimate_tube1d
from src.tvulog import ExtractorVisualizer1D
from src.tvulog.differential_operators import ScaleNormalizedLaplacian1D
from src.ulog.gaussian_scale_space import scale_space_representation1d
from src.tvulog.util import scale_discretization
from ...util import ExampleProblem, PerformanceComparison, PerformanceComparisonDataHandler, TVULoGApplication, \
    TVULoGDataHandler, add_colorbar_to_axis
from ...plot_params import CW, CW2


MU1 = 1.       # too large
MU2 = .01   # just right
MU3 = 1e-6      # too small
ETA1 = 1.       # too large
ETA2 = .1      # just right
ETA3 = 0.001    # too small


class OnedimensionalDeconvolution(ExampleProblem):
    """
    Implements the one-dimensional deconvolution problem used in section 5.1 of the paper.
    """
    def __init__(self, out: Path, plots: Path, mode: str):
        """

        Parameters
        ----------
        out
            Directory where the computed output is saved.
        plots
            Directory where the generated plots are saved.
        mode
            Computation mode ("test", "base" or "final").
        """
        np.random.seed(44)  # Set seed for reproducibility.
        n = 200
        psf_param = n / 20
        gmrf_prec = 200  # precision for GMRF prior.
        noise_std = 0.030
        sigma_min = 2
        sigma_max = 70
        # choose a logarithmic discretization of the scale range, as suggested by Lindeberg (1998).
        k = 30
        num_scales = k + 1
        self._sigmas = scale_discretization(sigma_min, sigma_max, num_scales)
        # Credibility parameter alpha (alpha=0.05 for 95% credibility)
        self._alpha = 0.05
        # Get forward operator from CUQIpy.
        A, y_data, info = cuqi.testproblem.Deconvolution1D.get_components(dim=n, PSF_param=psf_param,
                                                                          noise_std=noise_std)
        # Create custom sinc ground truth.
        x_rad = 14
        x_span = -x_rad + np.arange(n) * 2 * x_rad / n
        x_true = np.divide(np.sin(x_span), x_span, out=np.ones_like(x_span), where=x_span != 0)
        #x_true = np.array(info.exactSolution)
        y_true = A(x_true)
        noise = noise_std * np.random.randn(y_true.size)
        y_data = y_true + noise
        x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), gmrf_prec)
        y = cuqi.distribution.Gaussian(A(x), noise_std ** 2)
        self._bp = cuqi.problem.BayesianProblem(y, x)
        self._bp.set_data(y=y_data)
        self._path_lb = out / "onedimconv_lb.npy"
        self._path_ub = out / "onedimconv_ub.npy"
        self._path_estimate = out / "onedimconv_est.npy"
        self._path_samples = out / "onedimconv_samples.npy"
        self._y_obs = y_data
        self._y_true = y_true
        self._mode = mode
        self._ground_truth = x_true
        self._out = out
        self._plots = plots
        self._signal_shape = (n, )
        ExampleProblem.__init__(self, out=out, plots=plots)

    def generate_samples(self, num_samples: int):
        """
        Generates posterior samples using Linear RTO. The generated samples are stored as .npy-file in the
        `out`-directory.

        Parameters
        ----------
        num_samples
            Desired number of samples.
        """
        posterior_samples = self._bp.sample_posterior(num_samples)
        samples = posterior_samples.samples.T
        samples = samples.reshape(num_samples, *self._signal_shape)
        np.save(str(self._path_samples), samples)

    def compute_credible_intervals(self):
        """
        Estimates the credible scale-space tube from the samples. This requires that `self.generate_samples` has been
        called previously. The estimated tube is given in form of a lower bound array `lb` and an upper bound array
        `ub`. These arrays are not returned, instead they are stored as .npy-files in the `out`-directory.
        """
        samples = np.load(str(self._path_samples))
        # Compute posterior mean.
        x_mean = np.mean(samples, axis=0)
        np.save(str(self._path_estimate), x_mean)
        # Compute filtered credible intervals from samples.
        lb, ub = estimate_tube1d(alpha=self._alpha, samples=samples, sigmas=self._sigmas,
                                 neglogpdf=self._neglogpdf, ref=x_mean)
        np.save(str(self._path_lb), lb)
        np.save(str(self._path_ub), ub)

    def plot_setup(self):
        """
        Creates the plots that visualize the "setup" of the deconvolution problem.
        The generated plots are saved in the `plots`-directory.
        """
        x_true = self._ground_truth
        x_est = self._get_estimate()
        ssr_est = scale_space_representation1d(x_est, sigmas=self._sigmas)
        y_true = self._y_true
        y_obs = self._y_obs

        # --- Plot of the deconvolution problem.
        fig, ax = plt.subplots(1, 3, figsize=(CW, 0.5 * CW), sharey=True)
        ax[0].plot(x_true)
        ax[0].set_title(r"$f^*$")
        ax[1].plot(y_true)
        ax[1].set_title(r"$G f^*$")
        ax[2].plot(y_obs)
        ax[2].set_title(r"$y$")
        plt.savefig(str(self._plots / "deconvolution_setup.png"), bbox_inches="tight")

        # --- Plot of the credible scale-space tube.
        # load credible intervals
        lb = np.load(str(self._path_lb))
        ub = np.load(str(self._path_ub))
        # Invert such that smallest scale is in last row.
        lower_stack = lb[::-1, :]
        upper_stack = ub[::-1, :]
        ssr_est = ssr_est[::-1, :]
        # Make plot.
        vmin = np.min(lower_stack)
        vmax = np.max(upper_stack)
        fig, ax = plt.subplots(1, 3, figsize=(CW, 0.3 * CW), sharey=True)
        ax[0].imshow(lower_stack, cmap="gnuplot", aspect="auto", vmin=vmin, vmax=vmax)
        ax[0].set_title(r"$u^\mathrm{low}$")
        ax[1].imshow(upper_stack, cmap="gnuplot", aspect="auto", vmin=vmin, vmax=vmax)
        ax[1].set_title(r"$u^\mathrm{upp}$")
        ax[2].imshow(ssr_est, cmap="gnuplot", aspect="auto", vmin=vmin, vmax=vmax)
        ax[2].set_title(r"$u^\mathrm{mean}$")
        plt.savefig(str(self._plots / "deconvolution_tube.png"), bbox_inches="tight")

        # -- Plot of a horizontal (fixed scale) slice through the tube.
        fig2, ax2 = plt.subplots(1, 1, figsize=(CW, CW))
        k = lower_stack.shape[0]
        index = 5 * int(k / 6)
        ax2.plot(lower_stack[index], label=r"$u^\mathrm{low}$", color="green", linestyle=":")
        ax2.plot(upper_stack[index], label=r"$u^\mathrm{upp}$", color="blue", linestyle=":")
        ax2.fill_between(np.arange(lower_stack.shape[1]), lower_stack[index], upper_stack[index], alpha=0.2)
        ax2.plot(ssr_est[index], label=r"$u^\mathrm{mean}$", color="red")
        ax2.legend()
        ax2.set_title(f"t={int(self._sigmas[-index] ** 2)}")
        plt.savefig(str(self._plots / "deconvolution_tube_slice.png"), bbox_inches="tight")

    def compute_performance_comparison(self):
        """
        Runs the three different optimization approaches (dual/primal smoothing and interior point) on the
        deconvolution example. The traces are stored in the `out`-directory.
        """
        # create data handler
        data_handler = PerformanceComparisonDataHandler(out=self._out)
        # load credible intervals
        lb = np.load(str(self._path_lb))
        ub = np.load(str(self._path_ub))
        performance = PerformanceComparison(lb=lb, ub=ub, sigmas=self._sigmas,
                                            width_to_height=1., data_handler=data_handler)
        # Set parameters.
        performance.primal_scale = 1e6
        performance.dual_scale = 1e6
        performance.mu1 = MU1
        performance.mu2 = MU2
        performance.mu3 = MU3
        performance.eta1 = ETA1
        performance.eta2 = ETA2
        if self._mode == "test":
            performance.dual_fpg_iter = 1000
            performance.primal_fpg_iter = 1000
            performance.primal_bfgs_iter = 1000
            performance.socp_iter = 2000
            performance.socp_large_tol = 1e5
            performance.socp_steps = [5, 10, 20]  # [20, 40, 80]
        elif self._mode == "base":
            performance.dual_fpg_iter = 100000
            performance.primal_fpg_iter = 100000
            performance.primal_bfgs_iter = 30000
            performance.socp_iter = 100
            performance.socp_steps = [5, 10, 15, 20, 40, 60, 80, 100]
        else:
            performance.dual_fpg_iter = 100000
            performance.primal_fpg_iter = 100000
            performance.primal_bfgs_iter = 50000
            performance.socp_iter = 100
            performance.socp_steps = [2, 5, 8, 10, 15, 20, 25, 30, 35, 40]
        # -- Run the performance tests.
        # Dual smoothing
        performance.dual_smoothing_fpg()
        # Primal smoothing
        performance.primal_smoothing_fpg()
        # Interior point.
        performance.socp()

    def compute_tvulog(self):
        """
        Solves the TV-ULoG optimization problem for the deconvolution problem.
        It is important that `self.compute_credible_intervals` was called beforehand.
        """
        # create data handler
        data_handler = TVULoGDataHandler(out=self._out)
        # load credible intervals
        lb = np.load(str(self._path_lb))
        ub = np.load(str(self._path_ub))
        method_comp = TVULoGApplication(sigmas=self._sigmas, width_to_height=1., data_handler=data_handler)
        # Set parameters.
        if self._mode == "test":
            method_comp.tv_ulog_max_iter = 30
            method_comp.tv_ulog_tol = 1e8
        else:
            method_comp.tv_ulog_max_iter = 100
            method_comp.tv_ulog_tol = 1e-3
        method_comp.at_ulog_gamma = 1e6
        method_comp.at_ulog_epsilon = 1e-6
        method_comp.at_ulog_maxiter = 1000
        method_comp.at_ulog_ftol = 1e-14
        method_comp.at_ulog_backend = "ipopt"

        # -- Run the method comparison.
        print("Run TV-ULoG")
        method_comp.tv_ulog(lb=lb, ub=ub, scaled=True)
        print("Method comparison successfully run.")

    def plot_performance_comparison(self):
        """
        Creates the plots that compare the performance of the different optimization approaches. The first plot
        compares dual/primal smoothing (for a single smoothing parameter) to interior-point. The second plot
        compares the dual/primal smoothing methods for different choices of smoothing parameters.
        Both plots are stored in the `plots`-directory.
        """
        e_min = 1e-4
        data_handler = PerformanceComparisonDataHandler(out=self._out)
        # load results.
        traj_fdpg1 = data_handler.load_trajectory(method="fdpg1")
        traj_fdpg2 = data_handler.load_trajectory(method="fdpg2")
        traj_fdpg3 = data_handler.load_trajectory(method="fdpg3")
        traj_fpg1 = data_handler.load_trajectory(method="fpg1")
        traj_fpg2 = data_handler.load_trajectory(method="fpg2")
        traj_fpg3 = data_handler.load_trajectory(method="fpg3")
        traj_socp = data_handler.load_trajectory(method="socp")
        ref = np.min(traj_socp[1]) - e_min

        def normalize(x):
            return (x - ref) / traj_fdpg1[1][0]

        # --- First plot: Compare dual/primal smoothing to interior-point approach.
        fig_x, ax_x = plt.subplots(1, 1, figsize=(CW, CW))
        ax_x.semilogy(traj_fdpg2[0], normalize(traj_fdpg2[1]), color="blue",
                      label=f"dual smoothing")
        ax_x.semilogy(traj_fpg2[0], normalize(traj_fpg1[1]), color="red", label=fr"primal smoothing")
        ax_x.semilogy(traj_socp[0], normalize(traj_socp[1]), color="green", label="interior point")
        ax_x.set_xlabel("time [s]")
        ax_x.set_ylabel("normalized objective")
        ax_x.set_ylim(e_min, None)
        ax_x.legend(prop={'size': 6})
        plt.savefig(str(self._plots / "deconvolution_optimization.png"), bbox_inches="tight")

        # --- Second plot: Investigate dependence on smoothing parameter.
        fig2, ax2 = plt.subplots(1, 1, figsize=(CW2, .5 * CW2))
        ax2.semilogy(traj_fdpg1[0], normalize(traj_fdpg1[1]), color="purple",
                     label=fr"dual smoothing, $\mu$ = {ETA1:.2g}")
        ax2.semilogy(traj_fdpg2[0], normalize(traj_fdpg2[1]), color="blue", label=f"dual smoothing, $\mu$ = {ETA2:.2g}")
        ax2.semilogy(traj_fdpg3[0], normalize(traj_fdpg3[1]), color="lightblue",
                     label=f"dual smoothing, $\mu$ = {ETA3:.2g}")
        ax2.semilogy(traj_fpg1[0], normalize(traj_fpg1[1]), color="darkred",
                     label=fr"primal smoothing, $\mu$ = {MU1:.2g}")
        ax2.semilogy(traj_fpg2[0], normalize(traj_fpg2[1]), color="red", label=f"primal smoothing, $\mu$ = {MU2:.2g}")
        ax2.semilogy(traj_fpg3[0], normalize(traj_fpg3[1]), color="orange",
                     label=f"primal smoothing, $\mu$ = {MU3:.2g}")
        ax2.legend()
        plt.savefig(str(self._plots / "deconvolution_smoothing_parameter.png"), bbox_inches="tight")

    def plot_tvulog(self):
        """
        Plots the results of the TV-ULoG method applied to the deconvolution problem.
        Creates two plots. The first plot shows the normalized Laplacian of the minimizer (with the posterior mean
        as reference) in scale space.
        The second plot visualizes the detected blob-regions in the one-dimensional signal domain.
        Both plots are stored in the `plots`-directory.
        """
        # Create data handler.
        data_handler = TVULoGDataHandler(out=self._out)
        # Load CTV minimizer and normalized Laplacian.
        minimizer = data_handler.load_result(name="tv_minimizer")
        normlap = data_handler.load_result(name="tv_normlap")
        # Load ground truth
        x_true = self._ground_truth
        x_est = self._get_estimate()
        ssr_true = scale_space_representation1d(x_true, sigmas=self._sigmas)
        ssr_est = scale_space_representation1d(x_est, sigmas=self._sigmas)
        normlap_operator = ScaleNormalizedLaplacian1D(size=x_true.size, sigmas=self._sigmas)
        normlap_true = normlap_operator.fwd_arr(ssr_true)
        normlap_est = normlap_operator.fwd_arr(ssr_est)
        # Extract blobs from normalized_laplacian.
        mode_sets, blob_sets = self._extract_blob_regions(normlap)
        # Re-order so that smallest scale is at bottom.
        normlap_true = normlap_true[::-1, :]
        normlap_est = normlap_est[::-1, :]
        minimizer = minimizer[::-1, :]
        normlap = normlap[::-1, :]

        # --- First plot: TV-ULoG minimizer in scale space.
        # Two rows, three columns.
        fig, ax = plt.subplots(2, 3, figsize=(CW2, 0.5 * CW2), sharey="row")
        # - First row: Normalized Laplacian in scale space.
        # First panel shows the normalized Laplacian of the posterior mean.
        ax[0, 0].imshow(normlap_est, cmap="gnuplot", aspect="auto", norm=self._power_norm(vmax=np.max(normlap_true),
                                                                                          vmin=np.min(normlap_true)))
        ax[0, 0].set_title(r"$\tilde{\Delta} u^\mathrm{mean}$")
        # Second panel shows the normalized Laplacian of the TV-ULoG minimizer.
        mappable = ax[0, 1].imshow(normlap, cmap="gnuplot", aspect="auto",
                                   norm=self._power_norm(vmax=np.max(normlap_true), vmin=np.min(normlap_true)))
        ax[0, 1].set_title(r"$\tilde{\Delta} \bar{u}$")
        # Third panel shows the extracted blob regions.
        mode_set_image = np.zeros_like(minimizer)
        for mode_set in mode_sets:
            mode_set_image[mode_set.indices] = 1.
        # Reorder so that smallest scale is at bottom (more intuitive visualization).
        mode_set_image = mode_set_image[::-1, :]
        ax[0, 2].imshow(mode_set_image, cmap="gnuplot", aspect="auto", norm=self._power_norm(vmax=np.max(normlap_true),
                                                                                             vmin=np.min(normlap_true)))
        ax[0, 2].set_title(r"extracted blob regions")
        # Add colorbar.
        add_colorbar_to_axis(fig, ax[0, 1], mappable, ticks=False)
        # - Second row: Slice for fixed scale.
        k = minimizer.shape[0]
        scale_index = int(k / 2)
        ax[1, 0].plot(normlap_est[scale_index])
        ax[1, 1].plot(normlap[scale_index])
        ax[1, 2].remove()
        # Store the plot.
        plt.savefig(str(self._plots / "deconvolution_normlap_comparison.png"), bbox_inches="tight")

        # --- Second plot: Visualize blob regions in the signal domain.
        # Load posterior mean and ground truth.
        x_mean = np.load(str(self._path_estimate))
        x_true = self._ground_truth
        # Make a plot of the posterior mean and ground truth, together with a visualization of the blob sets.
        fig, ax = plt.subplots(1, 1, figsize=(CW, CW))
        ax.plot(x_true, color="g", label=r"$f^*$", linestyle=":")
        ax.plot(x_mean, color="r", label=r"$f^\mathrm{mean}$")
        # Visualize blob sets by horizontal bars.
        for blob_inner, blob_outer in blob_sets:
            i_min_inner = np.min(blob_inner.indices[0])
            i_max_inner = np.max(blob_inner.indices[0])
            h = np.max(x_mean[i_min_inner:i_max_inner])
            # Larger errorbar first. Otherwise, inner errorbar get overwritten.
            i_min_outer = np.min(blob_outer.indices[0])
            i_max_outer = np.max(blob_outer.indices[0])
            i_mid_outer = 0.5 * (i_min_outer + i_max_outer)
            i_err_outer = 0.5 * (i_max_outer - i_min_outer)
            ebar = ax.errorbar(i_mid_outer, h, xerr=i_err_outer, color="steelblue", capsize=2)
            ebar[-1][0].set_linestyle(":")
            # Now inner error bar.
            i_mid_inner = 0.5 * (i_min_inner + i_max_inner)
            i_err_inner = 0.5 * (i_max_inner - i_min_inner)
            h = np.max(x_mean[i_min_inner:i_max_inner])
            ax.errorbar(i_mid_inner, h, xerr=i_err_inner, color="steelblue", capsize=5)
        ax.legend(loc="upper right")
        plt.savefig(str(self._plots / "deconvolution_blobs.png"), bbox_inches="tight")

    def _get_y_obs(self) -> np.ndarray:
        return self._y_obs

    def _get_estimate(self) -> np.ndarray:
        x_est = np.load(str(self._path_estimate))
        return x_est

    def _neglogpdf(self, x):
        """
        Evaluates the negative log-posterior density function on a given vector `x`.
        """
        return -float(self._bp.posterior.logpdf(x.flatten()))

    def _extract_blob_regions(self, normlap: np.ndarray):
        """
        Extracts the blob regions from the normalized Laplacian using the extraction procedure described in the paper.
        """
        rthresh = 0.8  # Threshold for plateau extraction. Higher means smaller plateaus.
        tv_ulog = ExtractorVisualizer1D(sigmas=self._sigmas)
        tv_ulog_result = tv_ulog.extract_blobs(normlap=normlap, rthresh=rthresh, maxima_thresh=0.01)
        return tv_ulog_result.mode_sets, tv_ulog_result.blob_sets

    @staticmethod
    def _power_norm(vmax, vmin=0.):
        return colors.PowerNorm(gamma=1, vmin=vmin, vmax=vmax)
