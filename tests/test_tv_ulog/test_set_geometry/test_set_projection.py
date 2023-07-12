
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

from src.tvulog.set_geometry._set_projection import set_projection, scale_projection, _circle_projection2d
from src.tvulog.set_geometry import extract_modes


PLOT = False


def test_set_projection_2d():
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant2d.npy")
    modes = extract_modes(arr=x, rthresh=0.6)
    sigmas = 1 + np.arange(x.shape[0])
    blob_sets = [set_projection(mode_set=mode, shape=(x.shape[1],)) for mode in modes]
    # Visualize modes.
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        # Visualize modes.
        mode_img = np.zeros_like(x)
        for mode in modes:
            mode_img[mode.indices] = 1.
        # Shade projected modes.
        for blob in blob_sets:
            ind = blob.indices
            mode_img[:, ind] += 0.5
        ax.imshow(mode_img)
        plt.show()


def test_scale_projection_2d():
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant2d.npy")
    modes = extract_modes(arr=x, rthresh=0.6)
    sigmas = 1 + np.arange(x.shape[0])
    blob_set_tuples = [scale_projection(mode_set=mode, sigmas=sigmas, shape=(x.shape[1],)) for mode in modes]
    # Visualize blob sets and modes.
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        # Visualize modes.
        mode_img = np.zeros_like(x)
        for mode in modes:
            mode_img[mode.indices] = 1.
        # Shade projected modes.
        for blob_centers, blob_scales in blob_set_tuples:
            # Visualize scale extend.
            ind_extend = blob_scales.indices
            mode_img[:, ind_extend] += 0.25
            # Visualize blob centers.
            ind_centers = blob_centers.indices
            mode_img[:, ind_centers] += 0.25
        ax.imshow(mode_img)
        plt.show()


def test_set_projection_3d():
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant3d.npy")
    x = - x
    x_stack = x.clip(min=0.)
    k = x_stack.shape[0]
    sigmas = 1. + 1.5 * np.arange(k)
    modes = extract_modes(arr=x_stack, rthresh=0.8)
    # blob sets
    blob_sets = [set_projection(mode_set=mode, shape=x[0].shape) for mode in modes]
    # Make indicator function for mode stack.
    mode_stack = np.zeros_like(x_stack)
    for mode in modes:
        mode_stack[mode.indices] = 1.
    for blob in blob_sets:
        blob_indices = blob.indices
        mode_stack[:, blob_indices[0], blob_indices[1]] += 0.5
    mmin = np.min(mode_stack)
    mmax = np.max(mode_stack)
    if PLOT:
        fig, ax = plt.subplots(k, 1, figsize=(8, 20))
        for i in range(k):
            ax[i].imshow(mode_stack[i], cmap="gnuplot", vmin=mmin, vmax=mmax)
        plt.tight_layout()
        plt.show()


def test_circle_projection2d():
    num_scales = 10
    m = 12
    n = 53
    i = 5
    j = 24
    k = 3
    width_to_height = 2.
    shape = (12, 53)
    sigmas = 1 + 0.5 * np.arange(num_scales)
    point = np.array([k, i, j])
    points_in_circle = _circle_projection2d(point=point, shape=shape, sigmas=sigmas, width_to_height=width_to_height)
    points_in_circle = tuple(points_in_circle.T)
    # Visualize.
    if PLOT:
        image = np.zeros((m, n))
        fig, ax = plt.subplots(1, 1)
        image[point[1], point[2]] += 1.
        image[points_in_circle] += 0.25
        ax.imshow(image)
        r = sqrt(2) * sigmas[3]
        ax.set_title(f"x = ({k}, {i}, {j}), r = {r:.2g}")
        plt.show()


def test_scale_projection_3d():
    x = np.load("test_tv_ulog/test_set_geometry/piecewise_constant3d.npy")
    x = - x
    x_stack = x.clip(min=0.)
    k = x_stack.shape[0]
    sigmas = 1. + 1.5 * np.arange(k)
    modes = extract_modes(arr=x_stack, rthresh=0.8)
    # blob sets
    blob_set_tuples = [scale_projection(mode_set=mode, sigmas=sigmas, shape=x[0].shape, width_to_height=1.)
                       for mode in modes]
    # Make indicator function for mode stack.
    mode_stack = np.zeros_like(x_stack)
    for mode in modes:
        mode_stack[mode.indices] = 1.
    for blob_centers, blob_shadow in blob_set_tuples:
        ind_center = blob_centers.indices
        mode_stack[:, ind_center[0], ind_center[1]] += 0.25
        ind_shadow = blob_shadow.indices
        mode_stack[:, ind_shadow[0], ind_shadow[1]] += 0.25
    mmin = np.min(mode_stack)
    mmax = np.max(mode_stack)
    if PLOT:
        fig, ax = plt.subplots(k, 1, figsize=(8, 20))
        for i in range(k):
            ax[i].imshow(mode_stack[i], cmap="gnuplot", vmin=mmin, vmax=mmax)
        plt.tight_layout()
        plt.show()