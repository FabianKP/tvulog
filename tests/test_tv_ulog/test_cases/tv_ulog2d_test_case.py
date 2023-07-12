
from matplotlib import pyplot as plt
import numpy as np
import os


dirname = os.path.dirname(__file__)
# Stored values, don't change!
WIDTH_TO_HEIGHT = 2.
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 1.5
SIGMAS = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]


def tv_ulog2d_test_case():
    """
    Test case for TV-ULoG optimization problem for two-dimensional images, i.e. for a problem of the form

    min_x ||D L x||_1 s. t. lb <= x <= ub,

    where `D` is the forward-difference operator and `L` is the scale-normalized Laplacian for 2D images.
    The test case is created using pre-computed FCIs from the uq4pk-paper.

    Returns
    -------
    lb : shape (K, M, N)
        The lower bound FCIs.
    ub : shape (K, M, N)
        The upper bound FCIs.
    map_image : shape (M, N)
        The reference solution (the MAP image).
    sigmas : list
        The used sigma-values.
    width_to_height : float
        The used width-to-height ratio.
    """
    lb_path = os.path.join(dirname, 'data/blob_detection_lower_stack.npy')
    ub_path = os.path.join(dirname, 'data/blob_detection_upper_stack.npy')
    map_path = os.path.join(dirname, 'data/blob_detection_map.npy')
    lb = np.load(str(lb_path))
    ub = np.load(str(ub_path))
    map_image = np.load(str(map_path))
    sigmas = SIGMAS
    width_to_height = WIDTH_TO_HEIGHT
    return lb, ub, map_image, sigmas, width_to_height


def test_tv_ulog2d_test_case():
    # Get test case.
    lb, ub, map_image, sigmas, width_to_height = tv_ulog2d_test_case()
    # Visualize.
    k = len(sigmas)
    fig, ax = plt.subplots(k, 2)
    for i in range(k):
        vmax = ub[i].max()
        ax[i, 0].imshow(lb[i], vmax=vmax)
        ax[i, 1].imshow(ub[i], vmax=vmax)
    plt.show()
