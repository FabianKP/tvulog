
import numpy as np


class ExperimentData:
    """
    Assembles all relevant parameters for an experiment.
    """
    name: str                   # A string used as identifier.
    hermite_order: int          # The order of the Gauss-Hermite expansion. For example, ``hermite_order = 3``
                                # corresponds to a Gauss-Hermite expansion with 4 degrees of freedom (h_0, ..., h_3).
    snr: float                  # The signal-to-noise ratio of the measurement.
    y: np.ndarray               # The MASKED measurement.
    y_sd: np.ndarray            # The vector of standard errors for the MASKED measurement.
    f_ref: np.ndarray           # A reference distribution function. This might be a good guess, or the ground truth
                                # in the case where the data is simulated.
    theta_ref: np.ndarray       # Reference values for the theta_v hyperparameter.
    theta_guess: np.ndarray     # An 'initial guess' for the hyperparameter theta_v.
    theta_sd: np.ndarray        # Prior standard deviations associated to the initial guess on theta_v.
    mask: np.ndarray            # A Boolean mask for the measurement. If mask[i]=1, then _y[i] is included in the
                                # inference. If mask[i]=0, it is ignored.
