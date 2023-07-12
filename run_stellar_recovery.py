
from pathlib import Path
from examples.stellar_recovery.src import StellarRecovery


EXAMPLE_FOLDER = Path("examples/stellar_recovery")

mode = "final"   # "test" just for checking that code runs, "base" for faster computations,
                # "final" for final computations.

if mode == "test":
    out = Path(EXAMPLE_FOLDER / "out/out_test")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_test")
elif mode == "base":
    out = Path(EXAMPLE_FOLDER / "out/out_base")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_base")
elif mode == "final":
    out = Path(EXAMPLE_FOLDER / "out/out_final")
    plots = Path(EXAMPLE_FOLDER / "plots/plots_final")
else:
    raise ValueError("Only available modes are 'test', 'base' and 'final'.")

# Create folders if they do not exist.
out.mkdir(parents=True, exist_ok=True)
plots.mkdir(parents=True, exist_ok=True)

# Create example.
example = StellarRecovery(out=out, plots=plots, mode=mode)
# Generate MCMC samples.
#example.generate_samples()
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