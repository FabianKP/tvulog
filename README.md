# Uncertainty Quantification for Scale-Space Blob Detection

This repository contains the Python code that accompanies the paper

*Uncertainty Quantification for Scale-Space Blob Detection* ([arxiv-preprint]()).

In particular, it contains the code that was used to produce all figures in the paper.

## Authors

- Fabian Parzer (University of Vienna)
- Otmar Scherzer (University of Vienna, RICAM Linz, CD Lab "MaMSi")

## Contents

This repository is structures as follows:

- `examples/`: Contains the code for the numerical experiments in section 5 of the paper. 
- `src/`: Contains the implementation of the main routines.
- `tests/`: Contains unit tests written in `pytest` for the most important modules.

The figures can be reproduced with by running the scripts
- `run_one_dimensional_deconvolution.py`: For the deconvolution example (section 5.1/5.3 of the paper),
- `run_stellar_recovery.py`: For the stellar-recovery-example (section 5.2/5.3) of the paper.

## Contact

If you have any questions regarding this repository, feel free to write an email to
[fabian.kai.parzer@univie.ac.at](mailto:fabian.kai.parzer@univie.ac.at).
