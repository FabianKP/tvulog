
from jax import random
from matplotlib import pyplot as plt
plt.style.use("examples/tv_ulog.mplstyle")
from matplotlib import colors
import numpy as np
from pathlib import Path

from examples.stellar_recovery.src.visualization import plot_ulog_blobs, plot_tvulog_blobs, plot_distribution_function
from src.tvulog import ExtractorVisualizer2D
from src.tvulog.util import scale_discretization
from src.ulog import ULoG
from ...util import ExampleProblem, PerformanceComparison, PerformanceComparisonDataHandler, TVULoGApplication, \
    TVULoGDataHandler
from ...plot_params import CW
from examples.stellar_recovery.src.uq4pk_fit.stellar_model import StatModel, LightWeightedForwardOperator
from src.ulog.scale_space_tube import estimate_tube2d
from src.uq4pk_src import model_grids
from examples.stellar_recovery.src.mock import load_experiment_data
from ._parameters import DATA, DV, LMD_MIN, LMD_MAX


MU1 = 1e-3
MU2 = 1e-5
MU3 = 1e-8
ETA1 = 100.
ETA2 = 1.
ETA3 = 1e-2


class StellarRecovery(ExampleProblem):
    """
    Implements the numerical example "Integrated-light stellar population recovery" discussed in section 5.2 of the
    paper.
    """
    def __init__(self, out: Path, plots: Path, mode: str):
        """

        Parameters
        ----------
        out
            Directory for storing the computed output.
        plots
            Directory for storing the created plots.
        mode
            Computation mode ("test", "base" or "final").
        """
        # Load experiment data.
        data = load_experiment_data(DATA)
        # Store mode, which is used throughout.
        self._mode = mode
        # Parameters for scale-space representation.
        self._width_to_height = 2.
        sigma_min = .5
        sigma_max = 30.
        k = 15
        num_scales = k + 1
        self._sigmas = scale_discretization(sigma_min, sigma_max, num_scales)
        # Parameters for stellar model.
        theta_true = data.theta_ref
        y_sum = np.sum(data.y)
        y = data.y / y_sum
        sigma_y = data.y_sd / y_sum
        # Create ssps-grid.
        ssps = model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
        ssps.logarithmically_resample(dv=DV)
        self._ssps = ssps
        # Setup regularization term.
        snr = np.linalg.norm(y) / np.linalg.norm(sigma_y)
        self._regfactor = 500
        self._rthresh1 = 0.02
        self._rthresh2 = self._rthresh1
        self._overlap1 = 0.5
        self._overlap2 = 0.5
        regularization_parameter = self._regfactor * snr
        # Create `StatModel` instance.
        forward_operator = LightWeightedForwardOperator(theta=theta_true, ssps=ssps, do_log_resample=True)
        self._stellar_model = StatModel(y=data.y, y_sd=data.y_sd, forward_operator=forward_operator)
        self._stellar_model.beta = regularization_parameter
        f_map = self._stellar_model.compute_map()
        f_true = data.f_true.reshape(f_map.shape)
        self._alpha = 0.05
        self._f_true = f_true
        self._f_map = f_map
        # Set path variables.
        self._path_estimate = out / "stellar_recovery_f_map.npy"
        self._path_lb = out / "stellar_recovery_lb.npy"
        self._path_ub = out / "stellar_recovery_ub.npy"
        self._path_gradient_lb = out / "stellar_recovery_gradient_lb.npy"
        self._path_gradient_ub = out / "stellar_recovery_gradient_ub.npy"
        self._path_samples = out / "stellar_recovery_samples.npy"
        # Call constructor of super class.
        ExampleProblem.__init__(self, out=out, plots=plots)

    def generate_samples(self, *args, **kwargs):
        """
        Generates posterior samples using SVD-MCMC.
        """
        if self._mode == "test":
            burnin = 50
            num_samples = 100
        elif self._mode == "base":
            burnin = 1000
            num_samples = 5000
        else:
            burnin = 5000
            num_samples = 10000
        # Set RNG key for reproducibility
        rng_key = random.PRNGKey(32743)
        # Sample beta_tilde
        beta_array = self._stellar_model.sample_posterior(num_warmup=burnin, num_samples=num_samples, q=15,
                                                          method="svd", rng_key=rng_key)
        np.save(str(self._path_samples), beta_array)

    def compute_credible_intervals(self):
        """
        Estimates the credible scale-space tube from the samples. This requires that `self.generate_samples` has been
        called previously. The estimated tube is given in form of a lower bound array `lb` and an upper bound array
        `ub`. These arrays are not returned, instead they are stored as .npy-files in the `out`-directory.
        """
        samples = np.load(str(self._path_samples))
        # Append MAP estimate to samples to guarantee that it lies inside.
        extended_samples = np.zeros((samples.shape[0] + 1, samples.shape[1], samples.shape[2]))
        extended_samples[:samples.shape[0], :, :] = samples
        extended_samples[samples.shape[0], :, :] = self._f_map[:, :]
        lower_stack, upper_stack = estimate_tube2d(alpha=self._alpha, samples=extended_samples, sigmas=self._sigmas,
                                                   width_to_height=self._width_to_height, neglogpdf=self._neglogpdf,
                                                   ref_image=self._f_map)
        np.save(str(self._path_lb), lower_stack)
        np.save(str(self._path_ub), upper_stack)

    def plot_setup(self):
        """
        There are no plots for the imaging setup.
        """
        pass

    def compute_performance_comparison(self):
        """
        Runs the three different optimization approaches (dual/primal smoothing and interior point) on the
        stellar recovery example. The traces are stored in the `out`-directory.
        """
        # create data handler
        data_handler = PerformanceComparisonDataHandler(out=self._out)
        # load credible intervals
        lb = np.load(str(self._path_lb))
        ub = np.load(str(self._path_ub))
        performance = PerformanceComparison(lb=lb, ub=ub, sigmas=self._sigmas,
                                            width_to_height=self._width_to_height, data_handler=data_handler)
        # Set parameters.
        t_max = self._sigmas[-1]
        performance.primal_scale = 128 * (t_max ** 2)   # 1e-7 ?
        print(f"primal stepsize = {1. / performance.primal_scale}")
        performance.dual_scale = 128 * (t_max ** 2)
        performance.mu1 = MU1
        performance.mu2 = MU2
        performance.mu3 = MU3
        performance.eta1 = ETA1
        performance.eta2 = ETA2
        performance.eta3 = ETA3
        performance.dual_fpg_restart = False
        if self._mode == "test":
            performance.dual_fpg_iter = 1000
            performance.primal_fpg_iter = 1000
            performance.socp_iter = 2000
            performance.socp_steps = [5, 10, 20]  # [20, 40, 80]
        elif self._mode == "base":
            performance.dual_fpg_iter = 200000
            performance.primal_fpg_iter = 200000
            performance.socp_iter = 100
            performance.socp_steps = [10, 20, 40, 60, 100]  # [20, 40, 80]
        else:
            performance.dual_fpg_iter = 200000
            performance.primal_fpg_iter = 200000
            performance.socp_iter = 100
            performance.socp_steps = 5 * np.arange(1, 21)
        # -- Run the performance tests.
        # Dual smoothing
        performance.dual_smoothing_fpg()
        # Primal smoothing
        performance.primal_smoothing_fpg()
        # Interior point
        performance.socp()

    def compute_tvulog(self):
        """
        Solves the TV-ULoG optimization problem for the stellar recovery problem.
        It is important that `self.compute_credible_intervals` was called beforehand.
        """
        # create data handler
        data_handler = TVULoGDataHandler(out=self._out)
        # load credible intervals
        lb = np.load(str(self._path_lb))
        ub = np.load(str(self._path_ub))
        applier = TVULoGApplication(sigmas=self._sigmas, width_to_height=1., data_handler=data_handler)
        # Set parameters.
        if self._mode == "test":
            applier.tv_ulog_max_iter = 10
            applier.tv_ulog_tol = 1e8
        elif self._mode == "base":
            applier.tv_ulog_max_iter = 100
            applier.tv_ulog_tol = 1e-02
        else:
            applier.tv_ulog_max_iter = 150
            applier.tv_ulog_tol = 1e-02
        # -- Apply ULoG and TV-ULoG and store the results.
        print("Running ULoG.")
        applier.ulog(lb=lb, ub=ub)
        print("Running TV-ULoG.")
        applier.tv_ulog(lb=lb, ub=ub, scaled=True)
        print("Method comparison successfully run.")

    def plot_performance_comparison(self):
        """
        Creates the plots that compare the performance of the different optimization approaches. The first plot
        compares dual/primal smoothing (for a single smoothing parameter) to interior-point. The second plot
        compares the dual/primal smoothing methods for different choices of smoothing parameters.
        Both plots are stored in the `plots`-directory.
        """
        e_min = 1e-5
        # Load precomputed results.
        data_handler = PerformanceComparisonDataHandler(out=self._out)
        traj_fdpg1 = data_handler.load_trajectory(method="fdpg1")
        traj_fdpg2 = data_handler.load_trajectory(method="fdpg2")
        traj_fdpg3 = data_handler.load_trajectory(method="fdpg3")
        traj_fpg1 = data_handler.load_trajectory(method="fpg1")
        traj_fpg2 = data_handler.load_trajectory(method="fpg2")
        traj_fpg3 = data_handler.load_trajectory(method="fpg3")
        traj_socp = data_handler.load_trajectory(method="socp")
        ref = np.min(np.concatenate([traj_fpg1[1], traj_fpg2[1], traj_socp[1]])) - e_min

        def normalize(x):
            return (x - ref) / traj_fdpg1[1][0]
        # --- First plot: Compare dual/primal smoothing to interior-point approach.
        fig_x, ax_x = plt.subplots(1, 1, figsize=(CW, CW))
        ax_x.semilogy(traj_fdpg2[0], normalize(traj_fdpg2[1]), color="blue", label=f"dual smoothing")
        ax_x.semilogy(traj_fpg2[0], normalize(traj_fpg2[1]), color="red", label=f"primal smoothing")
        ax_x.semilogy(traj_socp[0], normalize(traj_socp[1]), color="green", label="interior point")
        ax_x.set_xlabel("time [s]")
        ax_x.set_ylabel("normalized objective")
        ax_x.set_ylim(e_min, None)
        ax_x.legend(prop={'size': 6})
        plt.savefig(str(self._plots / "stellar_recovery_optimization.png"), bbox_inches="tight")

        # --- Second plot: Investigate dependence on smoothing parameter.
        fig2, ax2 = plt.subplots(1, 1, figsize=(CW, CW))
        ax2.semilogy(traj_fdpg1[0], normalize(traj_fdpg1[1]), color="purple",
                     label=fr"dual smoothing, $\mu$ = {ETA1:.2g}")
        ax2.semilogy(traj_fdpg2[0], normalize(traj_fdpg2[1]), color="blue",
                     label=f"dual smoothing, $\mu$ = {ETA2:.2g}")
        ax2.semilogy(traj_fdpg3[0], normalize(traj_fdpg3[1]), color="lightblue",
                     label=f"dual smoothing, $\mu$ = {ETA3:.2g}")
        ax2.semilogy(traj_fpg1[0], normalize(traj_fpg1[1]), color="darkred",
                     label=fr"primal smoothing, $\mu$ = {MU1:.2g}")
        ax2.semilogy(traj_fpg2[0], normalize(traj_fpg2[1]), color="red",
                     label=f"primal smoothing, $\mu$ = {MU2:.2g}",)
        ax2.semilogy(traj_fpg3[0], normalize(traj_fpg3[1]), color="orange",
                     label=f"primal smoothing, $\mu$ = {MU3:.2g}")
        ax2.legend(prop={'size': 6})
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("normalized objective")
        ax2.set_ylim(1e-4, 1e4)
        plt.savefig(str(self._plots / "stellar_recovery_smoothing_parameter.png"), bbox_inches="tight")

    def plot_tvulog(self):
        """
        Plots the results of the TV-ULoG method applied to the stellar recovery problem.
        Creates a single plot that compares the results of TV-ULoG to the ground truth and the results of ULoG.
        """
        # Create data handler.
        data_handler = TVULoGDataHandler(out=self._out)
        # Load pre-computed results.
        ulog_blanket = data_handler.load_result(name="ulog_minimizer")
        tv_ulog_normlap = data_handler.load_result(name="tv_normlap")

        # Make the plot.
        fig, ax = plt.subplots(3, 1, figsize=(CW, 2 * CW))
        f_true = self._f_true
        # Top panel: Ground truth
        plot_distribution_function(ax=ax[0], image=f_true, ssps=self._ssps, flip=False, ylabel=True)
        # Middle panel: Visualization of TV-ULoG results.
        im1 = self._plot_tvulog(ax[1], normlap=tv_ulog_normlap)
        # Bottom panel: Visualization of ULoG results.
        im2 = self._plot_ulog(ax[2], blanket=ulog_blanket)
        # Add colorbar to each axis.
        for axis in ax:
            cbar_width = axis.get_position().height / 10
            cax = fig.add_axes([axis.get_position().x1 + 0.01, axis.get_position().y0, cbar_width,
                                axis.get_position().height])
            cbar = plt.colorbar(im1, cax=cax, aspect=10)
        # Set titles.
        ax[0].set_title("Ground Truth")
        ax[1].set_title("TV-ULoG")
        ax[2].set_title("ULoG")
        plt.savefig(str(self._plots / "stellar_recovery_tvulog_vs_ulog.png"), bbox_inches="tight")
        self._plot_tvulog_minimizer()

    def _get_estimate(self) -> np.ndarray:
        x_est = np.load(str(self._path_estimate))
        return x_est

    def _neglogpdf(self, x):
        return self._stellar_model.negative_log_pdf(x)

    def _apply_tv_ulog(self, normlap: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def _power_norm(vmax, vmin=0.):
        return colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)

    def _plot_tvulog_minimizer(self):
        # Create data handler.
        data_handler = TVULoGDataHandler(out=self._out)
        tv_ulog_normlap = data_handler.load_result(name="tv_normlap")
        vmax2 = np.max(tv_ulog_normlap)
        vmin2 = np.min(tv_ulog_normlap)
        k = len(self._sigmas)
        k_half = int(k / 2)
        fig, ax = plt.subplots(k_half, 1, figsize=(2 * CW, 3 * CW))
        for i in range(k_half):
            normlap_i = tv_ulog_normlap[i]
            plot_distribution_function(ax=ax[i], image=normlap_i, ssps=self._ssps, flip=False, vmax=vmax2, vmin=vmin2,
                                       xlabel=False)
        plt.savefig(str(self._plots / "stellar_recovery_tvulog_normlap.png"), bbox_inches="tight")

    def _plot_ulog(self, ax: plt.Axes, blanket: np.ndarray, xlabel: bool = True, ylabel: bool = True):
        """
        Auxiliary function that extracts the ULoG-blobs from a minimizer of the ULoG optimization problem.
        """
        # Load precomputed arrays
        f_map = self._f_map
        # Perform uncertainty-aware blob detection.
        ulog = ULoG(sigmas=self._sigmas, width_to_height=self._width_to_height)
        ulog_result = ulog.extract_blobs(minimizer=blanket, reference=f_map, rthresh1=self._rthresh1,
                                         rthresh2=self._rthresh2, overlap1=self._overlap1, overlap2=self._overlap2)
        # Create plot.
        im = plot_ulog_blobs(ax=ax, image=f_map, blobs=ulog_result.mapped_pairs, ssps=self._ssps, flip=False,
                             xlabel=xlabel, ylabel=ylabel)
        # Return mappable since this is required if one wants to add a colorbar.
        return im

    def _plot_tvulog(self, ax: plt.Axes, normlap: np.ndarray, xlabel: bool = True, ylabel: bool = True):
        """
        Extracts the blob regions from the normalized Laplacian and visualizes them in the image domain.
        """
        # Load precomputed arrays
        f_map = self._f_map
        # Extract blob regions
        tvulog = ExtractorVisualizer2D(sigmas=self._sigmas, width_to_height=self._width_to_height)
        tvulog_blobs = tvulog.extract_blobs(normlap=normlap, rthresh=0.5, maxima_thresh=0.01)
        # Visualize TV-ULoG result.
        im = plot_tvulog_blobs(ax=ax, image=f_map, blobs=tvulog_blobs.blob_sets, ssps=self._ssps, flip=False,
                               xlabel=xlabel, ylabel=ylabel)
        # Return mappable since this is required if one wants to add a colorbar.
        return im
