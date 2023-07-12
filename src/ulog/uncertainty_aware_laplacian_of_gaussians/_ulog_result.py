
from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence

from ..plotting import plot_scaled_ellipse


class ULoGResult:
    """
    Container for result of ULoG method.
    """
    def __init__(self, image: np.array, mapped_pairs: Sequence):
        self._image = image
        self.mapped_pairs = mapped_pairs

    def plot(self, color_inside: str = "lime", color_outside: str = "red", color_localization: str = "dodgerblue"):
        """
        Visualizes the result of the blob detection.

        Parameters
        ---
        color_inside
            Color for a significant blob.
        color_outside
            Color for a not-significant blob.
        color_localization
            Color for the localization radius.

        Returns
        ---
        fig
            A `matplotlib`-figure.
        ax
            A `matplotlib`-axis.
        """
        # Check that every element of blobs has length 2.
        for blob_tuple in self.mapped_pairs:
            assert len(blob_tuple) == 2
        # Plot image.
        fig, ax = plt.subplots()
        ax.imshow(self._image, cmap="gnuplot")
        # Plot blobs
        for blob_pair in self.mapped_pairs:
            b = blob_pair[0]
            c = blob_pair[1]
            if c is None:
                # Blob is not significant.
                y, x = b.position
                plot_scaled_ellipse(ax=ax, imshape=self._image.shape, center=(x, y), width=b.width, height=b.height,
                                    color=color_outside)
            else:
                # Blob is not significant.
                y1, x1 = b.position
                y2, x2 = c.position
                # If the width and height of the significant blob agree with reference blob, we increase the former
                # slightly for better visualization.
                factor = 1.05
                plot_scaled_ellipse(ax=ax, imshape=self._image.shape, center=(x1, y1), width=b.width, height=b.height,
                                    color=color_inside)
                plot_scaled_ellipse(ax=ax, imshape=self._image.shape, center=(x2, y2), width=factor * c.width,
                                    height=factor * c.height, color=color_localization, linestyle="--")
        return fig, ax
