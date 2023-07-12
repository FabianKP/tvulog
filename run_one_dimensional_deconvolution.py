
from pathlib import Path
from examples.one_dimensional_deconvolution.src import OnedimensionalDeconvolution


EXAMPLE_FOLDER = Path("examples/one_dimensional_deconvolution")

mode = "final"   # "test" just for checking that code runs, "base" for faster computations, "final" for final computations.

if mode == "test":
    out = Path(EXAMPLE_FOLDER / "out/out_test")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_test")
    num_samples = 1000
elif mode == "base":
    out = Path(EXAMPLE_FOLDER / "out/out_base")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_base")
    num_samples = 5000
elif mode == "final":
    out = Path(EXAMPLE_FOLDER / "out/out_final")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_final")
    num_samples = 10000
else:
    raise ValueError("Only available modes are 'test', 'base' and 'final'.")

# Create folders if they do not exist.
out.mkdir(parents=True, exist_ok=True)
plots.mkdir(parents=True, exist_ok=True)

# Create example.
example = OnedimensionalDeconvolution(out=out, plots=plots, mode=mode)
# Generate MCMC samples.
#example.generate_samples(num_samples)
# Compute credible intervals.
#example.compute_credible_intervals()
# Plot setup.
example.plot_setup()
# Compute performance comparison.
example.compute_performance_comparison()
# Plot performance comparison.
example.plot_performance_comparison()
# Compute method comparison.
#example.compute_tvulog()
# Plot method comparison.
#example.plot_tvulog()