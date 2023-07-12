
from matplotlib import patches
from matplotlib.pyplot import Axes
from typing import Tuple

def plot_scaled_ellipse(ax: Axes, imshape: Tuple[int, int], center: Tuple[int, int], width: float, height: float,
                        color: str, linestyle: str = "-"):
    """
    Plots an ellipse using normalized coordinates. This is done in-place, modifying
    the provided axis object.

    Parameters
    ----------
    ax
        Matplotlib `Axes` object where the ellipse should be plotted.
    imshape
        The shape of the underlying image.
    center
        The center coordinates of the ellipse.
    width
        The width of the ellipse.
    height
        The height of the ellipse.
    color
        The color of the ellipse boundary.
    linestyle
        The line style of the ellipse boundary.
    """
    vsize, hsize = imshape
    # ncenter = ((center[0] + 0.5), (center[1] - 0.5)). # I thought this is correct, but it leads to shifted ellipses.
    ncenter = (center[0], center[1])
    nwidth = width
    nheight = height
    ax.add_patch(patches.Ellipse(ncenter, width=nwidth, height=nheight, color=color, fill=False, linestyle=linestyle))