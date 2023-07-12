
import numpy as np


def l1l2_norm(x: np.array, d: int) -> float:
    """
    Evaluates the l1-l2 norm of an (n, d)-matrix.

    .. math::
        ||X||_{1,2} = \\sum_{i=1}^n ||X[i]||_2.

    Parameters
    ----------
    x : shape (n, d) or shape (nd,)
        The input `X` as 2D-matrix or flattened in column-major order.

    Returns
    -------
    y :
        The value of :math:`||X||_{1,2}`.
    """
    if x.ndim == 1:
        # Reshape x in column-major ordering.
        x = np.reshape(x, (-1, d), order="F")
    # Compute vector of inner norms.
    inner_norms = np.linalg.norm(x, axis=1)
    # Sum over to get l1-l2 norm.
    y = np.sum(inner_norms)
    return float(y)