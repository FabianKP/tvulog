
import numpy as np


def truncation(v: np.array, lb: np.array, ub: np.array) -> np.array:
    """
    Implements the proximal operator associated to the constraint `lb <= v <= ub`. This is simple truncation.
    """
    v[v >= ub] = ub[v >= ub]
    v[v <= lb] = lb[v <= lb]
    return v


def block_ball_projection(x: np.array, rho: float, d: int) -> np.array:
    """
    Orthogonal projection onto the set

    .. math::
        \{ x \\in \\mathbb{R}^{n \\times d} \\text{ : } ||x[i]||_2 \\leq \\rho \\}.

    Parameters
    ----------
    x : shape (nd, )
        The array, flattened in column-major order.
    rho :
        The radius of the block-balls.
    d :
        The inner dimension.

    Returns
    -------
    y :
        The projection of `x`.
    """
    # Check dimension.
    assert x.size % d == 0
    n = int(x.size / d)
    # Reshape x into shape (n, d).
    y = np.reshape(x, (n, d), order="F")
    # Perform row-wise projections.
    row_norms = np.linalg.norm(y, axis=1)
    too_large = np.where(row_norms >= rho)
    y[too_large] = rho * y[too_large] / row_norms[too_large, np.newaxis]
    # Flatten and return.
    return y.flatten(order="F")