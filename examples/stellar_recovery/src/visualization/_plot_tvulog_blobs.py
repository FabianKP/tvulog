
import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import patches
import numpy as np
from typing import Sequence, Tuple

from src.uq4pk_src.model_grids import MilesSSP
from src.tvulog.set_geometry import BlobSet
from ._plot_distribution_function import plot_distribution_function

from ._params import COLOR_SIGNIFICANT


def plot_tvulog_blobs(ax: Axes, image: np.ndarray, blobs: Sequence[Tuple[BlobSet, BlobSet]],
                      vmax: float = None, ssps: MilesSSP = None, flip: bool = True, xlabel: bool = True,
                      ylabel: bool = True):
    """
    Makes a blob-plot for given image and blobs.

    Parameters
    ----------
    ax
        The axis object where the plot should be created.
    image : shape (m, dim)
        The reference image.
    blobs
        The blob sets identified by TV-ULoG.
    vmax
        Maximum intensity shown in plot.
    ssps
        The grid object needed for plotting.
    flip
        If True, the plotted image is upside down. This is True by default, since it is more correct from a
        physical point of view.
    xlabel
        Determines whether x-axis is labeled or not.
    ylabel
        Determines whether y-axis is labeled or not.

    Returns
    -------
    immap
        Mappable. Can be used to add a colorbar.
    """
    # Plot distribution function.
    immap = plot_distribution_function(ax=ax, image=image, ssps=ssps, vmax=vmax, flip=flip, xlabel=xlabel,
                                       ylabel=ylabel)

    for blob_set_tuple in blobs:
        _plot_single_blob(ax=ax, blob_set_tuple=blob_set_tuple, imshape=image.shape, flip=flip)
    return immap


def _plot_single_blob(ax: plt.Axes, blob_set_tuple: Tuple[BlobSet, BlobSet], imshape: Tuple[int, int], flip: bool):
    """
    Plots a single blob in the given image.
    """
    blob_set_inner, blob_set_outer = blob_set_tuple
    polygon_inner = _blob_to_polygon(blob_set_inner, imshape=imshape, flip=flip, linestyle="-")
    polygon_outer = _blob_to_polygon(blob_set_outer, imshape=imshape, flip=flip, linestyle="--")
    ax.add_patch(polygon_inner)
    ax.add_patch(polygon_outer)


def _blob_to_polygon(blob: BlobSet, imshape: Tuple[int, int], flip: bool, linestyle) -> patches.Patch:
    """
    Given a blob, creates a corresponding `matplotlib.patches.Polygon`.
    """
    # Detect outer pixels.
    blob_mask = np.zeros(blob.shape, dtype=np.uint8)
    blob_mask[blob.indices] = 255
    # Upscale the mask image by factor 10.
    m, n = blob_mask.shape
    blob_mask = cv2.resize(blob_mask, (10 * n, 10 * m), interpolation=cv2.INTER_AREA)
    # Detect contour.
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_coords = contours[0].reshape(-1, 2)
    # Downscale contours again.
    contour_coords = 0.1 * contour_coords
    # Rescale the coordinates.
    m, n = imshape
    vec = np.array([1. / n, 1. / m])
    if flip:
        contour_rescaled = contour_coords * vec[np.newaxis, :]
    else:
        contour_rescaled = contour_coords
        contour_rescaled[:, 1] = m - contour_coords[:, 1]
        contour_rescaled = contour_rescaled * vec[np.newaxis, :]
    polygon_patch = patches.Polygon(xy=contour_rescaled, closed=True, color=COLOR_SIGNIFICANT, fill=False,
                                    linewidth=1.5, linestyle=linestyle)
    return polygon_patch