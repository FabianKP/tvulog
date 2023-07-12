import numpy as np
from scipy.sparse import csr_matrix
from time import time
from typing import Sequence

from ...util import l1l2_norm
from ._solve_socp import solve_socp
from .._ctv_solution import CTVSolution


def solve_socp_timing(a: csr_matrix, lb: np.array, ub: np.array, iters: Sequence[int], x_start: np.ndarray,
                      tol: float = 1e-1, verbose: bool = False) -> CTVSolution:
    d = int(a.shape[0] / a.shape[1])
    obj_list = [l1l2_norm(a @ x_start.flatten(), d)]
    print(f"initial obj = {obj_list[0]}")
    t_list = [0.]
    x = x_start
    for n_it in iters:
        t0 = time()
        x = solve_socp(a=a, lb=lb, ub=ub, x_start=x_start, maxiters=n_it, tol=tol, verbose=verbose)
        obj_x = l1l2_norm(a @ x.flatten(), d)
        t = time() - t0
        t_list.append(t)
        obj_list.append(obj_x)
        print(f"obj = {obj_x}")

    trajectory = np.array([t_list, obj_list])
    solution = CTVSolution(x=x, info={"trajectory": trajectory})
    return solution





