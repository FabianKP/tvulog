
import numpy as np
from .._regularization_operator import RegularizationOperator
from ._null_operator import NullOperator
from ._scaled_operator import ScaledOperator


def scale_operator(regop: RegularizationOperator, alpha: float) -> RegularizationOperator:
    """
    Scales a regularization operator: Given a regularization operator :math:`P` and a constant :math:`\\alpha`,
    the new operator is :math:`\\sqrt{\\alpha} P`.
    If :math:`\\alpha` is close to 0, then the returned operator is a `NullOperator`. Otherwise, it is a `ScaledOperator`.

    Parameters
    ---
    regop
        The regularization operator `P` that is to be scaled.
    alpha
        A strictly positive number by which the regularization operator is scaled.

    Returns
    ---
    """
    if np.isclose(abs(alpha), 0.):  # if alpha=0
        scaled_operator = NullOperator(dim=regop.dim)
    else:
        scaled_operator = ScaledOperator(regop=regop, alpha=alpha)
    return scaled_operator
