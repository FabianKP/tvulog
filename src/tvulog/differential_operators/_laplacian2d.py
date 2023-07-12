
import numpy as np
from scipy.ndimage import correlate1d
import scipy.sparse as sspa
from scipy.sparse.linalg import LinearOperator

from ._differential_operator import DifferentialOperator


def laplacian2d(image: np.ndarray, width_to_height: float) -> np.ndarray:
    """
    Computes the Laplacian of a given image, assuming Neumann boundary conditions.

    Parameters
    ----------
    image : shape (m, n)
        Input image.
    width_to_height
        Weighting factor of horizontal versus vertical axis.

    Returns
    -------
    y : shape (m, n)
        The discrete Laplacian of `image`.
    """
    assert image.ndim == 2
    central_difference_weights = np.array([1., -2., 1.])
    g1 = correlate1d(image, axis=0, weights=central_difference_weights)
    g2 = correlate1d(image, axis=1, weights=central_difference_weights) / width_to_height
    return g1 + g2


class Laplacian2D(DifferentialOperator):
    """
    Implementation of the Laplace operator for one-dimensional signals as child of `DifferentialOperator`.
    It uses a central-difference approximation, assuming Neumann boundary conditions.
    The Laplacian is represented by a matrix, assuming that the image is flattened in row-major order.
    This leads to a sparse (mn, mn)-matrix of the form:
    [A, 2I, 0, ..., 0,  0, 0]
    [I,  A, I, ..., 0,  0, 0]
    [0,  I, A, ..., 0,  0, 0]
    ...
    [0,  0, 0, ..., I,  A, I]
    [0,  0, 0, ..., 0, 2I , I],
    where I is the (n, n)-identity matrix and A is an (n, n)-matrix of the form
    A = [-a,  2b,  0, ...,  0,  0]
        [ b, -a,  b, ...,  0,  0]
        [ 0,  b, -a, ...,  0,  0]
        ...
        [ 0,  0,  0, ...,  b,  0]
        [ 0,  0,  0, ...,  2b, -a],
    where a = 2 * (1 + 1/r^2) and b = 1 / r^2 (see below).
    """
    def __init__(self, m: int, n: int, width_to_height: float):
        """
        Parameters
        ----------
        m
            Number of image rows.
        n
            Number of image columns.
        width_to_height
            How the image is scaled before the Laplacian is applied. The image is transformed via
            f_r(y1, y2) := f(T(y1, y2)) where T(y1, y2) = (y1, y2 * r), where `r = width_to_height`.
            We then take :math:`\Delta_r f(x1, x2) := \Delta f_r(y1, y2)`, where `(x1, x2) = T(y1, y2)`.
            By the chain rule, we then have

             .. math::
                \Delta_r f(x1, x2) := \Delta f_r(y1, y2)
                           = \partial_{y1}^2 f_r(y1, y2) + \partial_{y2}^2 f_r(y1, y2)
                           = \partial_{x1}^2 f(T^(-1)(y1, y2)) + \partial_{x2}^2 f(T^(-1)(y1, y2)
                           = \partial_{x1}^2 f(x1, x2) + 1 / r^2 \partial_{x2}^2 f(x1, x2).
        """
        if width_to_height <= 0.:
            raise ValueError("'width_to_height' must be a positive number.")
        # Both output and input have shape (m, n).
        shape = (m, n)
        self._width_to_height = width_to_height
        # Set the a- and b-values (see class docs)
        self._a = 2 + 2 * (1 / width_to_height) ** 2
        self._b = (1 / width_to_height) ** 2
        self._m = m
        self._n = n
        DifferentialOperator.__init__(self, x_shape=shape, y_shape=shape)
        self._csr_matrix = self._assemble_csr_matrix()

    def fwd_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the Laplacian to a given image.
        Parameters
        ----------
        arr : shape (m, n)
            The image.
        Returns
        -------
        delta_arr : shape (m, n)
            The Laplacian of `arr`.
        """
        assert arr.shape == self.x_shape
        delta_arr = self.matvec(arr.flatten()).reshape(self.x_shape)
        return delta_arr

    @property
    def csr_matrix(self) -> sspa.csr_matrix:
        """
        Sparse representation of the two-dimensional (flattened) Laplacian.
        """
        return self._csr_matrix

    @property
    def m(self) -> int:
        """
        Number of rows of the underlying image domain.
        """
        return self._m

    @property
    def n(self) -> int:
        """
        Number of columns of the underlying image domain.
        """
        return self._n

    def _assemble_csr_matrix(self) -> sspa.csr_matrix:
        """
        Block-wise assembling of the laplace matrix as csr.
        This uses `sspa.diags` to assemble the block-tridiagonal matrix, with `format="csr"`
        ensuring that the output is a `sspa.csr_matrix`.
        """
        # Create main diagonal with -4`s.
        main_diag = - self._a * np.ones(self.x_size)
        # Upper side diagonal is 2b at entries 0, n+1, ...; 0 at entries n, 2*n, ...; and b at the rest.
        upper_diagonal = self._b * np.ones(self.x_size - 1)
        upper_diagonal[np.arange(0, self.x_size-1) % self.n == 0] = 2 * self._b
        upper_diagonal[np.arange(1, self.x_size) % self.n == 0] = 0
        # Lower side diagonal is 2b at entries n-1 2*n-1, ...; 0 at entries n, 2*n, ...; and b at the rest.
        lower_diagonal = self._b * np.ones(self.x_size - 1)
        lower_diagonal[np.arange(2, self.x_size + 1) % self.n == 0] = 2 * self._b
        lower_diagonal[np.arange(1, self.x_size) % self.n == 0] = 0
        # Upper block diagonal is [2I, I, ...]
        upper_block_diagonal = np.concatenate([2 * np.ones(self.n), np.ones(self.x_size - 2 * self.n)])
        # Lower block diagonal is [I, ..., I, 2I]
        lower_block_diagonal = np.concatenate([np.ones(self.x_size - 2 * self.n), 2 * np.ones(self.n)])
        csr_mat = sspa.diags([main_diag, upper_diagonal, lower_diagonal, upper_block_diagonal, lower_block_diagonal],
                             [0, 1, -1, self.n, -self.n], format="csr")
        return csr_mat

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Outputs the Laplacian applied to the given flattened image.
        """
        assert x.size == self.x_size
        return self.csr_matrix.dot(x)

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Outputs the adjoint Laplacian applied to the given flattened image.
        """
        assert y.size == self.y_size
        return self.csr_matrix.T.dot(y)
