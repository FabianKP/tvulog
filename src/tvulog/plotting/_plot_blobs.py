
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np
from typing import Sequence, Tuple

from ..set_geometry import BlobSet


def plot_blob_sets(ax: plt.Axes, blobs: Sequence[Tuple[BlobSet, BlobSet]], color: str):
    """
    Visualizes the TV-ULoG blobs in the given axis.

    Parameters
    ----------
    ax
        `matplotlib` axis object.
    blobs
        Sequence of tuples of `BlobSet` objects, as is returned by the TV-ULoG method.
    color
        The color with which the set boundaries are indicated in the plot.
    """
    # Sequentially plot each blob-set-pair.
    for blob_set_tuple in blobs:
        _plot_single_blob(ax, blob_set_tuple, color)


def _plot_single_blob(ax: plt.Axes, blob_set_tuple: Tuple[BlobSet, BlobSet], color):
    """
    Visualizes a single blob-set-pair in the given axis.
    """
    blob_set_inner, blob_set_outer = blob_set_tuple
    # First blob set (first projection) is plotted with solid boundary.
    polygon_inner = _blob_to_polygon(blob_set_inner, color, linestyle="-")
    # Second blob set (second projection) is plotted with dashed boundary.
    polygon_outer = _blob_to_polygon(blob_set_outer, color, linestyle="--")
    ax.add_patch(polygon_inner)
    ax.add_patch(polygon_outer)


def _blob_to_polygon(blob: BlobSet, color, linestyle) -> Patch:
    """
    Given a blob set, creates a corresponding `matplotlib.patches.Polygon` that indicates the boundaries of the
    blob set in the image domain.
    """
    # Make mask that indicates all pixels inside the blob set.
    blob_mask = np.zeros(blob.shape, dtype=np.uint8)
    blob_mask[blob.indices] = 255
    # Upscale the mask image by factor 10 so that the mask boundaries can be plotted at pixel boundaries
    # and not only at pixel centers.
    m, n = blob_mask.shape
    blob_mask = cv2.resize(blob_mask, (10 * n, 10 * m), interpolation=cv2.INTER_AREA)
    # Detect contour.
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_coords = contours[0].reshape(-1, 2)
    # Downscale contours again.
    contour_coords = 0.1 * contour_coords
    # Have to shift.
    shift = 0.5 * np.ones(2)
    contour_coords = contour_coords - shift[np.newaxis, :]
    # Return as Polygon object.
    polygon_patch = Polygon(xy=contour_coords, closed=True, color=color, fill=False, linewidth=3, linestyle=linestyle)
    return polygon_patch
