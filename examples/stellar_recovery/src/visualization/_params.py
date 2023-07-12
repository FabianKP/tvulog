
from matplotlib import colors


CMAP = "magma_r"    # earlier we used "gnuplot"
COLOR_BLOB = "red"
COLOR_SIGNIFICANT = "dodgerblue"
COLOR_OUTSIDE = "red"
COLOR_INSIDE = "lime"


def power_norm(vmax, vmin=0.):
    return colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
