
from typing import List

from .._regularization_operator import RegularizationOperator
from ._block_operator import BlockOperator
from ._null_operator import NullOperator


def make_block_operator(operator_list: List[RegularizationOperator]) -> RegularizationOperator:
    """
    Given a list of regularization operators, creates a block operator as a direct sum.
    The resulting operator might either be a `BlockOperator`, or a `NullOperator` if all operators
    in the list are of instances of `NullOperator`.

    Parameters
    ---
    operator_list
        List of operators that are to be combined.

    Returns
    ---
    block_operator
        The combined block operator. Instance of `RegularizationOperator`.
    """
    # Check whether all operators in the list are null.
    all_null = True
    for op in operator_list:
        if not isinstance(op, NullOperator):
            all_null = False
    # If yes, return a NullOperator of the right dimension.
    if all_null:
        combined_dim = 0
        for op in operator_list:
            combined_dim += op.dim
        block_operator = NullOperator(combined_dim)
    # If not, return a BlockOperator.
    else:
        block_operator = BlockOperator(operator_list)
    return block_operator
