

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..gaussian_scale_space._scale_normalized_laplacian import scaled_laplacian
from ..optimization import solve_bls


def compute_blanket(lb: np.ndarray, ub: np.ndarray, width_to_height: float):
    """
    Solves
    .. math::
        \\min_B ||\\tilde{\\Delta} B||_2^2 \\text{ s. t. } \\ell \leq B \leq u,

    where :math:`B \\in \\mathbb{R}^{m \\times n}`.

    Parameters
    ----------
    lb : shape (m, n)
        The lower bound :math:`\ell \in \mathbb{R}^{m \times n}`.
    ub : shape (m, n)
        The upper bound :math:`u \in \mathbb{R}^{m \times n}`.
    width_to_height : float
        The width-to-height ratio used in the scale-normalized Laplacian.
        See the documentation of `gaussian_scale_space._normalized_laplacian?.

    Returns
    -------
    blanket : shape (m, n)
        The minimizer :math:`B`.
    """
    assert lb.shape == ub.shape
    assert np.all(lb <= ub)
    # If ub > lb, then we can safely return the zero minimizer.
    if ub.min() > lb.max():
        blanket = np.ones(lb.shape) * lb.max()
        return blanket

    # Create scipy.sparse.LinearOperator that represents the scaled Laplacian.
    def forward(x):
        x_im = x.reshape(lb.shape)
        y = scaled_laplacian(x_im, width_to_height=width_to_height).flatten()
        return y

    def adjoint(u):
        u_im = u.reshape(lb.shape)
        x = scaled_laplacian(u_im, width_to_height=width_to_height).flatten()
        return x
    laplace_op = LinearOperator(shape=(lb.size, lb.size), matvec=forward, rmatvec=adjoint)
    # Rescale everything to prepare optimization.
    scale = ub.max()
    lbvec = lb.flatten() / scale
    ubvec = ub.flatten() / scale
    # Solve bounded least-squares problem.
    rhs = np.zeros(lb.size)
    x_min = solve_bls(a=laplace_op, b=rhs, lb=lbvec, ub=ubvec)
    # Bring minimizer back to the original scale.
    x_min = scale * x_min
    # Bring minimizer into the correct format.
    blanket = np.reshape(x_min, lb.shape)
    # Return the solution as two-dimensional numpy array.
    return blanket
