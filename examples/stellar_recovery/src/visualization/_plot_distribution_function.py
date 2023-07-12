
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from src.uq4pk_src.model_grids import MilesSSP
from ._params import power_norm, CMAP


def plot_distribution_function(ax: Axes, image: np.ndarray, ssps: MilesSSP = None, vmax: float = None,
                               vmin: float = 0., flip: bool = False, xlabel: bool = True, ylabel: bool = False,
                               xticklabels: bool = True, yticklabels: bool = True):
    """
    Plots the age-metallicity distribution function with a colorbar on the side that
    shows which color belongs to which value.

    Parameters
    ----------
    ax
        A matplotlib.axes.Axes object.
    image
        The age-metallicity distribution as 2-dimensional numpy array.
    ssps
        The SSPS grid.
    vmax
        The maximum intensity that should be plotted.
    flip
        If True, the plotted image is upside down. This is True by default, since it is more correct from a
        physical point of view.
    xlabel
        Determines whether x-axis is labeled or not.
    ylabel
        Determines whether y-axis is labeled or not.
    xticklabels
        Determines whether the ticks on the x-axis are labeled.
    yticklabels
        Determines whether the ticks on the y-axis are labeled.

    Returns
    ---
    immap
        Mappable. Can be used to create a colorbar separately.
    """
    if flip:
        f_im = np.flipud(image)
    else:
        f_im = image
    cmap = plt.get_cmap(CMAP)
    if vmax is None:
        vmax = image.max()
    # I want fixed aspect ratio to 6:2.5.
    aspect = 2.5 / 6.
    immap = ax.imshow(f_im, cmap=cmap, extent=(0, 1, 0, 1), aspect=aspect, norm=power_norm(vmin=vmin, vmax=vmax))
    if xlabel:
        ax.set_xlabel("Age [Gyr]")
    if ylabel:
        ax.set_ylabel("Metallicity [Z/H]")
    if ssps is not None:
        ticks = [ssps.t_ticks, ssps.z_ticks, ssps.img_t_ticks, ssps.img_z_ticks]
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        if xticklabels:
            ax.set_xticklabels(t_ticks)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(img_z_ticks)
        if yticklabels:
            ax.set_yticklabels(z_ticks)
        else:
            ax.set_yticklabels([])
    # Return "mappable" (allows colorbar creation).
    return immap
