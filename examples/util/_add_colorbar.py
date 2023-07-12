
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np


cbar_aspect = 40
nticks = 5          # Desired number of ticks in colorbar.


def add_colorbar_to_axis(fig: plt.Figure, ax: plt.Axes, im, ticks: bool = True):
    """
    Adds colorbar next to given axis.

    Parameters
    ---
    fig
        The figure object.
    ax
        The axis in `fig` to which the colorbar is to be added.
    im
        The mappable that defines the colorbar.
    """
    cbar_width = ax.get_position().height / cbar_aspect
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, cbar_width, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, aspect=cbar_aspect)
    if not ticks:
        cbar.set_ticks([])


def add_colorbar_to_plot(fig, axes, im):
    """
    Adds a colorbar to whole plot.

    Parameters
    ---
    fig
        The figure object.
    axes
        The figure axes.
    im
        The mappable that defines the colorbar.
    """
    cbar = fig.colorbar(im, ax=axes, shrink=0.4, aspect=cbar_aspect)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()