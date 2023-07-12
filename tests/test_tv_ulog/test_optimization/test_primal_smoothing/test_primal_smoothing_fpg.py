
from matplotlib import pyplot as plt
import numpy as np

from src.tvulog.optimization._primal_smoothing._primal_smoothing_fpg import primal_smoothing_fpg
from src.tvulog.differential_operators import ForwardDifference, Laplacian2D
from tests.test_tv_ulog.test_cases import noisy_sine, gaussian_test


np.random.seed(2211)
PLOT = False


def test_first_order_one_dimensional():
    n = 500
    mu = 1.
    gtol = 1e-8
    lb, truth, ub = noisy_sine(n)
    nabla = ForwardDifference(shape=(n,), width_to_height=1.)
    a_op = nabla
    sol = primal_smoothing_fpg(a=a_op, mu=mu, lb=lb, ub=ub, max_iter=5000)
    x = sol.x
    if PLOT:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(lb)
        ax[0].plot(ub)
        ax[0].plot(truth)
        ax[0].plot(x)
        ax[1].plot((nabla @ x)[1:-1])
        ax[1].plot((nabla @ truth)[1:-1])
        plt.show()
    assert x.shape == (n, )


def test_two_dimensional():
    n = 30
    lower2d, map2d, upper2d = gaussian_test(n=n)
    mu = 1. / n
    m, n = map2d.shape
    delta = Laplacian2D(m=m, n=n, width_to_height=2.)
    a_op = delta
    sol = primal_smoothing_fpg(a=a_op, mu=mu, lb=lower2d.flatten(), ub=upper2d.flatten(), stepsize=1 / 64,
                               max_iter=5000)
    x = sol.x.reshape(m, n)
    if PLOT:
        fig, ax = plt.subplots(1, 4)
        vmax = upper2d.max()
        ax[0].imshow(lower2d, vmax=vmax)
        ax[1].imshow(x, vmax=vmax)
        ax[2].imshow(map2d, vmax=vmax)
        ax[3].imshow(upper2d, vmax=vmax)
        i = 15
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(lower2d[i])
        ax2.plot(x[i])
        ax2.plot(map2d[i])
        ax2.plot(upper2d[i])
        plt.show()
    assert x.shape == (m, n)