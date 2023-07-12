
from math import sqrt
import numpy as np
from typing import Sequence, Tuple

from ._blob_set import BlobSet
from ._mode_set import ModeSet


def set_projection(mode_set: ModeSet, shape: Tuple[int, ...]) -> BlobSet:
    """
    Project a mode set to the signal domain.
    For example (in 3D-scale space) a point `(k,i,j)` in the mode set is projected to the point `(i,j)`
    in the image domain.

    Parameters
    ----------
    mode_set
        The mode set, in scale space.
    shape
        The shape of the signal domain.

    Returns
    -------
    blob_set
        The projected mode set. An object of type `BlobSet`.
    """
    dim = mode_set.dim
    assert dim in [2, 3], "Only implemented for `dim` in [2, 3]."
    # remove scale indices
    projected_indices = np.array(mode_set.indices[1:])
    # remove duplicate columns
    projected_indices = np.unique(projected_indices, axis=1)
    # convert back to tuple, so that we have nice indexing
    projected_indices = tuple(projected_indices)
    # create `BlobSet` instance.
    blob_set = BlobSet(shape=shape, indices=projected_indices)
    return blob_set


def scale_projection(mode_set: ModeSet, shape: Tuple[int, ...], sigmas: Sequence[float], width_to_height: float = None)\
        -> Tuple[BlobSet, BlobSet]:
    """
    Projects a set in scale space to the original image domain in a way that visualizes the geometry both in space
    and scale. See the Visualization-section in the paper for more explanation.

    Parameters
    ----------
    mode_set
        The set in scale space.
    shape
        The shape of the signal domain.
    sigmas
        The sigma values for the scale discretization.
    width_to_height:
        Ratio of blob width to blob height.

    Returns
    -------
    blob_centers: BlobSet
        Credible region for the centers of the possible blobs.
    blob_shadow: BlobSet
        Created by representing each point `(x,t)` in scale space by a circle of radius :math:`\sqrt{2 t}` and
        center `x`, and taking the union of these circles over all points in `mode_set`.
    """
    dim = mode_set.dim
    assert dim in [2, 3], "Only implemented for `dim` in [2, 3]."
    # Get blob centers using `set_projection`.
    blob_centers = set_projection(mode_set, shape)

    # --- Get circle-projection of mode_set:
    # For each point `(x, t)` in mode_set, compute all points that are within sqrt{2t} radius of `x`.
    # Take the union over all these points to get all the points that should be in `blob_scale`.
    if dim == 2:
        circle_points = _scale_project1d(mode_set, shape, sigmas)
        circle_points_tuple = tuple([circle_points])
    else:
        assert width_to_height is not None
        circle_points = _scale_project2d(mode_set, shape, sigmas, width_to_height)
        # `circle_points` is a set of d-tuples and has to be converted to a tuple on ndarrays.
        circle_points_arr = np.array(circle_points)
        circle_points_tuple = tuple(circle_points_arr.T)
    # Create a `BlobSet` object `blob_shadow` from this result.
    blob_shadows = BlobSet(shape=shape, indices=circle_points_tuple)
    # Return `blob_centers` and `blob_shadows`.
    return blob_centers, blob_shadows


def _scale_project1d(mode_set: ModeSet, shape: Tuple[int, ...], sigmas: Sequence[float]) -> np.ndarray:
    """
    Projects set in two-dimensional scale space to a one-dimensional set of indices.

    Parameters
    ----------
    mode_set
        The mode set in the two-dimensional scale space.
    shape
        The shape of the signal domain.
    sigmas
        The standard deviations for the scale-space representation.

    Returns
    -------
    projection :
        A numpy array of integers representing the indices inside the projected set.
    """
    # Minimum of the projection is `min(i - \sqrt(t_k)`, maximum is `max(i + \sqrt{t_k})`.
    ind_arr = np.array(mode_set.indices).T
    i_min_float = np.min([i - sigmas[k] for k, i in ind_arr])
    i_max_float = np.max([i + sigmas[k] for k, i in ind_arr])
    i_min = max(0, int(np.floor(i_min_float)))
    i_max = min(shape[0] - 1, int(np.ceil(i_max_float)))
    return np.arange(i_min, i_max + 1)


def _scale_project2d(mode_set: ModeSet, shape: Tuple[int, ...], sigmas: Sequence[float], width_to_height: float)\
        -> Sequence[np.ndarray]:
    """
    Projects sets in three-dimensional scale space on two-dimensional set of indices.

    Parameters
    ----------
    mode_set
        The mode set in the three-dimensional scale space.
    shape
        The shape of the image domain.
    sigmas
        The standard deviations for the scale-space representation.
    width_to_height
        The ratio between blob width and blob height.

    Returns
    -------
    projection :
        The projected set, represented as tuple of 2 numpy arrays. The first array contains the x1-indices of the points
        in the projection, the second contains the x2-indices.
    """
    # For each point of the circle, compute the corresponding indices in the circle.
    projection_list = []
    mode_set_points = np.array(mode_set.indices).T
    # Iterate over points in scale space.
    for point in mode_set_points:
        # For every point, plot the circle around that point.
        circle_around_point = _circle_projection2d(point=point, shape=shape, sigmas=sigmas,
                                                   width_to_height=width_to_height)
        projection_list.extend(circle_around_point)
    # Remove duplicates and return.
    projection = np.array(projection_list)
    projection = np.unique(projection, axis=0)
    return projection


def _circle_projection2d(point: np.ndarray, shape: Tuple[int, ...], sigmas: Sequence[float], width_to_height: float)\
        -> Sequence[np.ndarray]:
    """
    Given a point (x,t) in scale space, returns all points that are within `\sqrt{2t}` distance of `x`.
    Only implemented for `d`=2,3.

    Parameters
    ----------
    point: shape (d, )
        The point in scale space.
    shape
        The shape of the signal domain.
    sigmas
        The sigma-values for the discrete scales.
    width_to_height:
        Ratio of blob width to blob height.

    Returns
    -------
    points_inside_circle : set
        Returns a set of d-tuples, each corresponding to a point in the circle.
    """
    # Compute r = \sqrt{t_k}.
    k, i, j = point
    tk = sigmas[k] ** 2
    r1 = sqrt(2) * sigmas[k] / width_to_height    # Radius in x1-direction.
    # Create list of points in circle.
    points_inside_circle = list([])
    # --- Add every point (i', j') for which (i' - i)^2 + (j' - j)^2 <= ceil(r^2).
    # WLOG we only have to consider max(0, i - ceil(r)) <= i <= min(m, i + ceil(r)).
    m, n = shape
    r_floor = int(np.floor(r1))
    i_min = max(0, i - r_floor)
    i_max = min(m - 1, i + r_floor)
    # Iterate over row indices i'.
    for ip in range(i_min, i_max + 1):
        # Iterate over column indices j'.
        # A point (i', j') is inside the circle if |j' - j| <= ceil(\sqrt(r^2 - (i - i')^2)).
        d_square = max(0, 2 * tk - (width_to_height * (i - ip)) ** 2)
        max_dist = int(np.floor(sqrt(d_square)))
        j_min = max(0, j - max_dist)
        j_max = min(n - 1, j + max_dist)
        # Add to set.
        points_inside_circle.extend([np.array([ip, jp]) for jp in range(j_min, j_max + 1)])
    # Remove duplicates.
    points_inside_circle = np.array(points_inside_circle)
    points_inside_circle = np.unique(points_inside_circle, axis=0)
    return points_inside_circle
