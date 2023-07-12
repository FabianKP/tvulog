
from matplotlib.pyplot import Axes
import numpy as np
from typing import List

from ..gaussian_blob import GaussianBlob
from ..plotting import plot_scaled_ellipse


COLOR_BLOB = "red"


class LoGResult:
    """
    Container for result of Laplacian-of-Gaussians method.
    """
    def __init__(self, image: np.array, blobs: List[GaussianBlob]):
        self._image = image
        self.blobs = blobs

    def plot(self, ax : Axes, blob_color: str = "red"):
        """
        Visualizes the result of the blob detection.

        Parameters
        ---
        ax
            Axis object on which the blob should be plotted.
        blob_color
            Color of the plotted blob has.
        """
        for blob in self.blobs:
            y, x = blob.position
            plot_scaled_ellipse(ax=ax, imshape=self._image.shape, center=(x, y), width=blob.width, height=blob.height,
                                color=blob_color)