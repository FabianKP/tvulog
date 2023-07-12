
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.tvulog.set_geometry._set_projection import set_projection
from src.tvulog.set_geometry import extract_modes
from src.tvulog.plotting._plot_blobs import plot_blob_sets


PLOT = False


def test_plot_blobs():
    x = np.load("test_tv_ulog/test_plotting/piecewise_constant3d.npy")
    x = - x
    x_stack = x.clip(min=0.)
    k = x_stack.shape[0]
    sigmas = 1. + 1.5 * np.arange(k)
    modes = extract_modes(arr=x_stack, rthresh=0.8)
    # blob sets
    blob_sets = [set_projection(mode_set=mode, shape=x[0].shape) for mode in modes]

    if PLOT:
        # Visualize results.
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(x[0], cmap="gnuplot")
        # Mark the blob-pixels
        for blob in blob_sets:
            blob_pixels = np.array(blob.indices).T
            for pixel in blob_pixels:
                pixel_rectangle = Rectangle(xy=(pixel[1] - 0.5, pixel[0] - 0.5), width=1, height=1, alpha=0.5, color="green")
                ax.add_patch(pixel_rectangle)
        plot_blob_sets(ax=ax, blobs=blob_sets, color="green")
        # Add reference (0,0) pixel.
        ref_pixel = Rectangle(xy=(0, 0), width=1., height=1., color="cyan")
        ax.add_patch(ref_pixel)
        plt.show()
