
import numpy as np
from pathlib import Path
from typing import Tuple, Literal, Sequence

from src.tvulog.optimization import TVULoGProblem, TVULoGProblemScaled, DualSmoothingFPG, PrimalSmoothingFPG,\
    InteriorPointSolver


METHOD_NAMES = ["fdpg1", "fdpg2", "fdpg3", "fpg1", "fpg2", "fpg3", "bfgs1", "bfgs2", "bfgs3", "socp"]
METHOD_TYPE = Literal["fdpg1", "fdpg2", "fdpg3", "fpg1", "fpg2", "fpg3", "bfgs1", "bfgs2", "bfgs3", "socp"]


class PerformanceComparisonDataHandler:
    def __init__(self, out: Path):
        """
        Handles the reading and writing of results that are created by a `PerformanceComparison` instance.

        Parameters
        ----------
        out
        """
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self._path_lower_bound = Path(out / "pc_lower_bound.npy")
        self._path_upper_bound = Path(out / "pc_upper_bound.npy")
        self._path_ground_truth = Path(out / "pc_ground_truth.npy")
        self._path_data = Path(out / "pc_y_obs.npy")
        self._path_estimate = Path(out / "pc_estimate.npy")
        self._path_trajectory = {method_name: Path(out / f"trajectory_{method_name}.npy")
                                 for method_name in METHOD_NAMES}
        self._path_minimizer = {method_name: Path(out / f"minimizer_{method_name}.npy")
                                 for method_name in METHOD_NAMES}
        self._path_normlap = {method_name: Path(out / f"normlap_{method_name}.npy")
                                 for method_name in METHOD_NAMES}

    def save_lower_upper_bound(self, lb: np.ndarray, ub: np.ndarray):
        np.save(str(self._path_lower_bound), lb)
        np.save(str(self._path_upper_bound), ub)

    def load_lower_upper_bound(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.load(str(self._path_lower_bound))
        ub = np.load(str(self._path_upper_bound))
        return lb, ub

    def save_ground_truth(self, truth: np.ndarray):
        np.save(str(self._path_ground_truth), truth)

    def load_ground_truth(self) -> np.ndarray:
        ground_truth = np.load(str(self._path_ground_truth))
        return ground_truth

    def load_data(self) -> np.ndarray:
        y_obs = np.load(str(self._path_data))
        return y_obs

    def save_data(self, y_obs: np.ndarray):
        np.save(str(self._path_data), y_obs)

    def load_estimate(self) -> np.ndarray:
        x_est = np.load(str(self._path_data))
        return x_est

    def save_estimate(self, x_est: np.ndarray):
        np.save(str(self._path_data), x_est)

    def save_trajectory(self, trajectory: np.ndarray, method: METHOD_TYPE):
        np.save(str(self._path_trajectory[method]), trajectory)

    def load_trajectory(self, method: METHOD_TYPE) -> np.ndarray:
        trajectory = np.load(str(self._path_trajectory[method]))
        return trajectory

    def save_normlap(self, normlap: np.ndarray, method: METHOD_TYPE):
        np.save(str(self._path_normlap[method]), normlap)

    def load_normlap(self, method: METHOD_TYPE):
        normlap = np.load(str(self._path_normlap[method]))
        return normlap

    def save_minimizer(self, minimizer: np.ndarray, method: METHOD_TYPE):
        np.save(str(self._path_minimizer[method]), minimizer)

    def load_minimizer(self, method: METHOD_TYPE):
        minimizer = np.load(str(self._path_minimizer[method]))
        return minimizer


class PerformanceComparison:

    def __init__(self, lb: np.ndarray, ub: np.ndarray, sigmas: Sequence[float], width_to_height: float,
                 data_handler: PerformanceComparisonDataHandler):
        # Create CTVProblem.
        tv_ulog_problem = TVULoGProblemScaled(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height)
        self.ctv_problem = tv_ulog_problem
        # Set other tunable parameters.
        self.dual_fpg_iter = 100000
        self.primal_fpg_iter = 100000
        self.primal_bfgs_iter = 30000
        self.socp_iter = 100
        self.socp_steps = [5, 10, 15, 20, 40, 60, 80, 100]
        self.primal_scale = 1.
        self.dual_scale = 1.
        self.mu1 = 1.
        self.mu2 = 0.1
        self.mu3 = 0.01
        self.eta1 = 1.
        self.eta2 = 0.1
        self.eta3 = 0.01
        self.verbose = True
        self.socp_large_tol = 1000
        # Define x_start as mean of bounds.
        self.x_start = tv_ulog_problem._rescale(0.5 * (lb + ub))
        self._handler = data_handler
        problem_unscaled = TVULoGProblem(lb=lb, ub=ub, sigmas=sigmas, width_to_height=width_to_height)
        self._delta_norm = problem_unscaled.delta_norm

    def dual_smoothing_fpg(self):
        for i, eta in zip(range(1, 4), [self.eta1, self.eta2, self.eta3]):
            print(f"Dual FPG with mu = {eta} ...")
            dual_smoothing = DualSmoothingFPG(stepsize=1. / self.dual_scale, mu=eta, max_iter=self.dual_fpg_iter)
            solution = dual_smoothing.solve_ctv_problem(self.ctv_problem, x_start=self.x_start)
            trajectory = solution.info["trajectory"]
            minimizer = self.ctv_problem._rescale_back(solution.x)
            x_normlap = self.ctv_problem.delta_norm.fwd_arr(minimizer)
            self._handler.save_trajectory(trajectory, method=f"fdpg{i}")
            self._handler.save_minimizer(minimizer, method=f"fdpg{i}")
            self._handler.save_normlap(x_normlap, method=f"fdpg{i}")


    def primal_smoothing_fpg(self):
        for i, mu in zip(range(1, 4), [self.mu1, self.mu2, self.mu3]):
            print(f"Primal FPG with mu = {mu} ...")
            print()
            primal_fpg = PrimalSmoothingFPG(stepsize=1. / self.primal_scale, mu=mu, max_iter=self.primal_fpg_iter)
            solution = primal_fpg.solve_ctv_problem(self.ctv_problem, x_start=self.x_start)
            trajectory = solution.info["trajectory"]
            minimizer = self.ctv_problem._rescale_back(solution.x)
            x_normlap = self._delta_norm.fwd_arr(minimizer)
            self._handler.save_trajectory(trajectory, method=f"fpg{i}")
            self._handler.save_minimizer(minimizer, method=f"fpg{i}")
            self._handler.save_normlap(x_normlap, method=f"fpg{i}")

    def socp(self):
        socp_solver = InteriorPointSolver(max_iter=self.socp_iter, verbose=self.verbose)
        solution = socp_solver.solve_with_timing(problem=self.ctv_problem, iterations=self.socp_steps,
                                                   large_tol=self.socp_large_tol, x_start=self.x_start)
        minimizer = self.ctv_problem._rescale_back(solution.x)
        trajectory = solution.info["trajectory"]
        x_normlap = self._delta_norm.fwd_arr(minimizer)
        self._handler.save_trajectory(trajectory, method="socp")
        self._handler.save_minimizer(minimizer, method="socp")
        self._handler.save_normlap(x_normlap, method="socp")
