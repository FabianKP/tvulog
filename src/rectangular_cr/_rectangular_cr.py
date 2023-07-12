
import numpy as np
from typing import Union


def rectangular_cr(theta: float, samples: np.array, g: Union[callable, np.array], mode: np.array,
                   transform: callable = None, max_iter: int = 1000, verbose: bool = True):
    """
    Computes rectangular credible regions from samples, using the method described in appendix B of the paper.
    The density values have to be provided so that the samples can be ordered.

    Parameters
    ----------
    theta : float
        The desired crediblity level. The function will return a box that contains at least `theta * 100` % of
        the samples.
    samples : shape (n, d)
        The d-dimensional samples, stacked row-wise.
    g : callable or np.array of shape (n, ).
        Either a  function g such that the target probability density is proportional to exp(-g(x))
        (e.g. the negative log-density), or a numpy-vector of the pre-computed values of g.
        If a function is provided, it should take numpy-arrays of shape (d, ) as input and return a floating point
        number `y` that satisfies `- np.inf < y <= np.inf`.
    mode : shape (d, )
        The mode (i.e. maximizer) of p.
    transform: optional, callable
        The user has the option to provide a function that takes arrays of shape (d, ) as arguments and output arrays
        of shape (m, ).  The credible region is then computed for the transformed samples.
    max_iter: optional, int
        Maximum number of bisection steps.
    verbose : bool
        If True, the method will print information to the console.

    Returns
    -------
    lb : shape (m, )
        The lower bound of the rectangular credible region. If `transform = None`, then m=d. Otherwise, `m` is
        the dimension of the output of the transform.
    ub : shape (m, )
        The upper bound of the rectangular credible region. If `transform = None`, then m=d. Otherwise, `m` is
        the dimension of the output of the transform.
    theta_est : float
        The achieved credibility level, i.e. the number of samples inside the estimated interval divided by
        the overall number of samples.
    """
    # Check input
    if not np.isscalar(theta):
        raise TypeError("'theta' must be a scalar.")
    if not 0. < theta < 1.:
        raise ValueError("'theta' must be between 0 and 1.")
    if not samples.ndim == 2:
        raise ValueError("'samples' must be a numpy array of dimension 2.")
    n = samples.shape[0]
    # Pre-sort samples in ascending distance from the mode.
    distances = np.linalg.norm(samples - mode[np.newaxis, :], axis=1)
    dist_order = np.argsort(distances)
    samples_sorted = samples[dist_order]
    # Get values of g.
    if isinstance(g, np.ndarray):
        g_vals = g[dist_order]
    else:
        # Evaluate g on each sample.
        g_vals = np.apply_along_axis(g, 1, samples_sorted)
    # Sort in ascending g-order. Because we sorted wrt distance, ties are decided by the distance to the mode.
    g_order = np.argsort(g_vals)
    samples_sorted = samples_sorted[g_order]
    # If a transform is provided, the samples are transformed.
    if transform is not None:
        samples_sorted = [transform(x).flatten() for x in samples_sorted]
    # Compute smallest axis-aligned box that contains the first n_theta samples.
    n_theta = np.ceil(theta * n).astype(int)
    box = _smallest_bounding_box(samples_sorted[:n_theta])
    # Count samples in box. If the number of contained samples is n_theta, we are done.
    n_inside = _count_inside(box, samples_sorted)
    # Otherwise, find the smallest number k such that the box that contains the first k samples contains exactly
    # n_theta samples. This is done via bisection.
    if n_inside != n_theta:
        k_min = 1
        k_max = n_theta
        count = 0
        k = n_theta
        while (n_inside != n_theta) and (count < max_iter):
            if n_inside > n_theta:
                k_max = k
            elif n_inside < n_theta:
                k_min = k
            k_old = k
            k = np.ceil(0.5 * (k_min + k_max)).astype(int)
            if k == k_old:
                break
            box = _smallest_bounding_box(samples_sorted[:k])
            n_inside = _count_inside(box, samples_sorted)
            count += 1
        if verbose: print(f"Bisection terminated after {count} steps.")
    theta_est = n_inside / n
    ub, lb = box

    return lb, ub, theta_est


def _smallest_bounding_box(points: np.array) -> np.array:
    """
    Computes the smallest axis-aligned box that contains the given points.

    Parameters
    ---
    points : shape (n, d)
        Array of `n` points in dimension `d`.

    Returns
    ---
    ub, lb : `ub` and `lb` are vectors of shape (d, ) that correspond to the upper and lower bound of the box.
    """
    lb = np.min(points, axis=0)
    ub = np.max(points, axis=0)
    return np.row_stack([ub, lb])


def _count_inside(box: np.array, points: np.array) -> int:
    """
    Counts how many points are inside the given box.

    Parameters
    ---
    box : shape (2, d)
        The first row should be the upper bound, the second row the lower bound of the box.
    points : shape (n, d)
        Array of `n` points in dimension `d`.

    Returns
    ---
    int : The number of vectors in `points` that are inside the given box.
    """
    ub, lb = box
    samples_inside = [x for x in points if np.all(x <= ub) and np.all(x >= lb)]
    num_inside = len(samples_inside)
    return num_inside