
from pathlib import Path


class ExampleProblem:
    """
    Abstract base class for numerical examples.
    """
    def __init__(self, out: Path, plots: Path):
        self._out = out
        self._plots = plots

    def generate_samples(self, *args, **kwargs):
        """
        Generate posterior samples necessary for the computation of credible intervals.
        """
        raise NotImplementedError

    def compute_credible_intervals(self, *args, **kwargs):
        """
        Computes and saves the filtered credible intervals.
        """
        raise NotImplementedError

    def compute_gradient_credible_intervals(self, *args, **kwargs):
        """
        Compute credible intervals for the gradient. Required for the contact formulation.
        """
        raise NotImplementedError

    def plot_setup(self):
        raise NotImplementedError

    def compute_performance_comparison(self):
        raise NotImplementedError

    def compute_tvulog(self):
        raise NotImplementedError

    def plot_performance_comparison(self):
        raise NotImplementedError

    def plot_tvulog(self):
        raise NotImplementedError