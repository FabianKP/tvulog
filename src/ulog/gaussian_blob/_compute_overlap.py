
import shapely.affinity as aff
import shapely.geometry as geom

from ._gaussian_blob import GaussianBlob


def compute_overlap(blob1: GaussianBlob, blob2: GaussianBlob) -> float:
    """
    Computes the relative overlap of two blobs, i.e.

    .. math::
        o_r = \\frac{A_{intersection}}{\\min(A_1, A_2)}.

    The implementation uses shapely (https://pypi.org/project/Shapely/).

    Parameters
    ----------
    blob1:
        The first blob.
    blob2:
        The second blob.

    Returns
    -------
    relative_overlap : float
        The relative overlap, a number between 0 and 1.
    """
    # Create shapely.ellipse objects
    ell1 = _create_ellipse(blob1)
    ell2 = _create_ellipse(blob2)
    # Compute areas of the two ellipses.
    a1 = ell1.area
    a2 = ell2.area
    # Compute intersection area.
    a_intersection = ell1.intersection(ell2).area
    # Compute relative overlap.
    relative_overlap = a_intersection / min(a1, a2)
    return relative_overlap


def _create_ellipse(blob: GaussianBlob):
    """
    Creates a shapely-ellipse object from a Gaussian blob.
    """
    circ = geom.Point(blob.position).buffer(1)
    ellipse = aff.scale(circ, 0.5 * blob.height, 0.5 * blob.width)
    return ellipse
