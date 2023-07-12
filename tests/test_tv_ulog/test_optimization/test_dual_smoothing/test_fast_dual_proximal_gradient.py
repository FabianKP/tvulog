
from matplotlib import pyplot as plt
import numpy as np

from src.tvulog.optimization._dual_smoothing._dual_smoothing import dual_smoothing_fpg
from src.tvulog.differential_operators import ForwardDifference
from tests.test_tv_ulog.test_cases import noisy_sine


def test_first_order():
    n = 500
    eps = 1e-4
    eta = 1e-6
    ftol = 4 * eps
    lb, truth, ub = noisy_sine(n)
    x_start = 0.5 * (lb + ub)
    a_op = ForwardDifference(shape=(n, ), width_to_height=1.)
    a_mat = np.array([a_op @ e_i for e_i in np.eye(n)]).T
    max_eig = np.linalg.norm(a_mat.T @ a_mat, ord=2)
    sol_fdpg = dual_smoothing_fpg(a=a_op, lb=lb, ub=ub, stepsize=1 / max_eig, x_start=x_start, mu=eta, max_iter=5000)
    x_fdpg = sol_fdpg.x