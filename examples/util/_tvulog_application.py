
import numpy as np
from pathlib import Path
from typing import Literal, Sequence, Tuple

from src.tvulog import tv_ulog, _tv_ulog_scaled
from src.ulog import compute_blanket


RESULT_NAMES = ["ulog_minimizer", "tv_minimizer", "tv_normlap"]
RESULT_TYPE = Literal["ulog_minimizer", "tv_minimizer", "tv_normlap"]


class TVULoGDataHandler:
    def __init__(self, out: Path):
        """
        Handles the reading and writing of results that are created by a `PerformanceComparison` instance.

        Parameters
        ----------
        out
        """
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self._path_lower_bound = Path(out / "mc_lower_bound.npy")
        self._path_upper_bound = Path(out / "mc_upper_bound.npy")
        self._path_ground_truth = Path(out / "mc_ground_truth.npy")
        self._path_to_result = {result_name: Path(out / f"{result_name}.npy") for result_name in RESULT_NAMES}

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

    def save_result(self, result: np.ndarray, name: RESULT_TYPE):
        np.save(str(self._path_to_result[name]), result)

    def load_result(self, name: RESULT_TYPE) -> np.ndarray:
        result = np.load(str(self._path_to_result[name]))
        return result


class TVULoGApplication:
    """
    Class that manages the application of TV-ULoG to an example problem.
    """
    def __init__(self, sigmas: Sequence[float], width_to_height: float,
                 data_handler: TVULoGDataHandler):
        self._sigmas = sigmas
        self._width_to_height = width_to_height
        self._handler = data_handler
        # Tunable parameters.
        self.tv_ulog_max_iter = 100
        self.tv_ulog_tol = 1e-4
        self.at_ulog_gamma = 1.
        self.at_ulog_epsilon = 0.1
        self.at_ulog_maxiter = 100
        self.at_ulog_ftol = 1e-10

    def ulog(self, lb: np.ndarray, ub: np.ndarray):
        """
        Solves the ULoG optimization problem::

            minimize ||normalized_laplacian(L)||
            subject to lb <= L <= ub.

        The computed minimizer is stored in the `out`-directory.

        Parameters
        ----------
        lb
            Lower bound of scale-space tube.
        ub
            Upper bound of scale-space tube.
        """
        # Apply ULoG.
        blanket_list = []
        for lb_i, ub_i in zip(lb, ub):
            blanket_i = compute_blanket(lb=lb_i, ub=ub_i, width_to_height=self._width_to_height)
            blanket_list.append(blanket_i)
        blankets = np.array(blanket_list)
        self._handler.save_result(blankets, name="ulog_minimizer")

    def tv_ulog(self, lb: np.ndarray, ub: np.ndarray, scaled: bool = True):
        """
        Solves the TV-ULoG optimization problem::

            minimize TV(normalized_laplacian(L))
            subject to lb <= L <= ub.

        The computed minimizer and its normalized Laplacian are stored in the `out`-directory.
        """
        # Apply TV-ULoG.
        options = {"max_iter": self.tv_ulog_max_iter, "tol": self.tv_ulog_tol}
        if scaled:
            tv_ulog_result = _tv_ulog_scaled(lb=lb, ub=ub, sigmas=self._sigmas, width_to_height=self._width_to_height,
                                             options=options)
        else:
            tv_ulog_result = tv_ulog(lb=lb, ub=ub, sigmas=self._sigmas, width_to_height=self._width_to_height,
                                     options=options)
        # Store minimizer and normalized Laplacian.
        self._handler.save_result(tv_ulog_result.minimizer, name="tv_minimizer")
        self._handler.save_result(tv_ulog_result.normlap, name="tv_normlap")
