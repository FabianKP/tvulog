
import numpy as np
import scipy.sparse as sspa
from typing import Tuple

from ._differential_operator import DifferentialOperator


def forward_difference(arr: np.ndarray, axis: int,  width_to_height: float = 1.) -> np.ndarray:
    """
    Computes the forward difference of the array `arr`` along the specified axis. For example, the forward difference
    of a 3-dimensional array along the second axis is a 3-dimensional array `y` of the same shape, given by
    `y[i,j,k] = x[i,j+1,k] - x[i,j,k]`, if `j+1` is in the range. If `j+1` is out of range, set `y[i,j,k] = 0`.

    Parameters
    ----------
    arr
        N-dimensional array.
    axis
        The axis along which the forward difference is to be computed.

    Returns
    -------
    y
        Array of the same shape as `arr`, where each entry gives the corresponding forward difference along the axis
        `axis`.
    """
    ndim = arr.ndim
    if ndim > 3:
        raise ValueError("Only for dimensions 1-3!")
    forward_diff = np.diff(arr, axis=axis, n=1)
    # Pad with zero along axis of differentiation.
    pad_widths = [(0, 0), ] * ndim
    pad_widths[axis] = (0, 1)
    forward_diff = np.pad(forward_diff, pad_width=pad_widths, mode="constant")
    # Apply width_to_height.
    if (ndim == 2 and axis == 1) or (ndim == 3 and axis == 2):
        forward_diff = forward_diff / width_to_height
    assert forward_diff.shape == arr.shape
    return forward_diff


def gradient_norms(x: np.ndarray, width_to_height: float = 1.) -> np.ndarray:
    """
    Convenience function for computing the vector :math:`||(\\nabla x)_i||_2`.

    Parameters
    ----------
    x : shape (N, )
        N-dimensional array or flattened vector.

    Returns
    -------
    tv_x : shape `x.shape`
        Vector of gradient norms.
    """
    nabla_list = [forward_difference(x, i, width_to_height) for i in range(x.ndim)]
    nabla_arr = np.array(nabla_list)
    # Now rows correspond to derivatives along axes.
    tv_x = np.linalg.norm(nabla_arr, axis=0).reshape(x.shape)
    return tv_x


class ForwardDifference(DifferentialOperator):
    """
    The discrete forward difference operator as child of `DifferentialOperator`.
    For higher-dimensional arrays, assumes that the image is flattened in row-major order.
    Only implemented for one-, two- and three-dimensional domains.
    """
    def __init__(self, shape: Tuple[int, ...], width_to_height: float):
        """
        Parameters
        ----------
        shape
            Shape of the signal domain.
        width_to_height
            A weighting between the second and third axis (only if `len(shape)==3`. Useful for the scale-space case.
        """
        if len(shape) > 3:
            raise NotImplementedError("Forward difference is only implemented up to dimension 3.")
        y_shape = (len(shape), *shape)
        DifferentialOperator.__init__(self, x_shape=shape, y_shape=y_shape)
        # Process width to height.
        self.width_to_height = width_to_height
        # Create csr_mat.
        if self.x_dim == 1:
            self._csr_matrices = self._csr_matrix1d()
        elif self.x_dim == 2:
            self._csr_matrices = self._csr_matrix2d()
        elif self.x_dim == 3:
            self._csr_matrices = self._csr_matrix3d()
        else:
            raise NotImplementedError
        self._csr_matrix = sspa.vstack(self._csr_matrices)

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Maps array to stacked forward difference. For example, array of shape (m, n) is mapped to (2, m, n).
        """
        fdiff = self.matvec(arr.flatten())
        fdiff = np.reshape(fdiff, self.y_shape, order="C")
        return fdiff

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        Returns matrix representation of forward difference operator along all axes. The output is a single
        `scipy.sparse.csr_matrix` instance.
        """
        return self._csr_matrix

    @property
    def csr_matrix_list(self) -> list[sspa.csr_matrix]:
        """
        The list of the matrices corresponding to the forward difference along the individual axes. Each element is
        a `scipy.sparse.csr_matrix` instance.
        """
        return self._csr_matrices

    def _csr_matrix1d(self) -> list[csr_matrix]:
        """
        Computes the representation as `scipy.sparse.csr_matrix` for one-dimensional domains.
        """
        x = np.concatenate([-np.ones(self.x_size - 1), np.ones(self.x_size - 1)])
        row_ind = np.concatenate([np.arange(self.x_size - 1), np.arange(self.x_size - 1)])
        col_ind = np.concatenate([np.arange(self.x_size - 1), np.arange(self.x_size - 1) + 1])
        csr_mat = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.y_size))
        return [csr_mat]

    def _csr_matrix2d(self) -> list[csr_matrix]:
        """
        Computes the representations as `scipy.sparse.csr_matrix` for two-dimensional domains. The individual
        derivatives are returned as separate matrix-objects.
        """
        csr_mat_list = []
        # Create derivative operator along direction 0.
        x = []
        row_ind = []
        col_ind = []
        for i in range(self.x_size):
            # Derivative in direction 0.
            if i + self.x_shape[1] < self.x_size:
                x.extend([-1., 1])
                row_ind.extend([i, i])
                col_ind.extend([i, i + self.x_shape[1]])
        csr_along_0 = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        csr_mat_list.append(csr_along_0)
        # Create derivative operator along direction 1.
        x = []
        row_ind = []
        col_ind = []
        for i in range(self.x_size):
            if (i + 1) % self.x_shape[1] != 0:
                x.extend([-1., 1])
                row_ind.extend([i, i])
                col_ind.extend([i, i + 1])
        csr_along_1 = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        # Rescale according to width-to-height.
        csr_along_1 = csr_along_1 / self.width_to_height
        csr_mat_list.append(csr_along_1)
        return csr_mat_list

    def _csr_matrix3d(self) -> list[csr_matrix]:
        """
        Computes the representations as `scipy.sparse.csr_matrix` for three-dimensional domains. The individual
        derivatives are returned as separate matrix-objects.
        """
        csr_mat_list = []
        # Create derivative operator along direction 0.
        x = []
        row_ind = []
        col_ind = []
        offset_0 = self.x_shape[1] * self.x_shape[2]
        for s in range(self.x_size):
            # Derivative in direction 0.
            k = s // (self.x_shape[1] * self.x_shape[2])
            not_at_boundary = (k < self.x_shape[0] - 1)
            if not_at_boundary:
                x.extend([-1., 1])
                row_ind.extend([s, s])
                col_ind.extend([s, s + offset_0])
        csr_along_0 = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        csr_mat_list.append(csr_along_0)
        # Create derivative operator along direction 1.
        x = []
        row_ind = []
        col_ind = []
        offset_1 = self.x_shape[2]
        for s in range(self.x_size):
            # Index along axis 1 must not be at boundary.
            i = (s // self.x_shape[2]) % self.x_shape[1]
            not_at_boundary = (i < self.x_shape[1] - 1)
            if not_at_boundary:
                x.extend([-1., 1])
                row_ind.extend([s, s])
                col_ind.extend([s, s + offset_1])
        csr_along_1 = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        csr_mat_list.append(csr_along_1)
        # Create derivative operator along direction 2.
        x = []
        row_ind = []
        col_ind = []
        offset_2 = 1
        for s in range(self.x_size):
            j = s % self.x_shape[2]
            not_at_boundary = (j < self.x_shape[2] - 1)
            if not_at_boundary:
                x.extend([-1., 1])
                row_ind.extend([s, s])
                col_ind.extend([s, s + offset_2])
        csr_along_2 = sspa.csr_matrix((x, (row_ind, col_ind)), shape=(self.x_size, self.x_size))
        # Rescale according to width-to-height.
        csr_along_2 = csr_along_2 / self.width_to_height
        csr_mat_list.append(csr_along_2)
        return csr_mat_list

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward differences for a given flattened array.

        Parameters
        ----------
        x : shape (n, )
            The array, flattened in row-major order.

        Returns
        -------
        nabla_x : shape (`self.im_dim` * n, )
            Returns the forward differences along each axis, stacked. That is, the first `x.size` entries are the
            forward difference along axis 0, the next `x.size` along axis 1, and so on.
        """
        assert x.size == self.x_size
        return self.csr_matrix @ x

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Implements the adjoint corresponding to `ForwardDifference._matvec`.
        Parameters
        ----------
        y : shape (d * n, )

        Returns
        -------
        x : shape (n, )
        """
        assert y.size == self.y_size
        return self.csr_matrix.T @ y