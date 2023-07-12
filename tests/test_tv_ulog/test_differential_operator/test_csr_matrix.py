
import numpy as np
import scipy.sparse as sspa

from src.tvulog.differential_operators._forward_difference import ForwardDifference


def test_recognize_nonzeros():
    m = 10
    n = 12
    v = ForwardDifference(shape=(m, n), width_to_height=1.)
    # Assemble gradient matrix.
    mat = np.array([v.matvec(e_i) for e_i in np.eye(v.shape[1])]).T
    # Have this as csr-matrix.
    csr_mat = sspa.csr_matrix(mat)
    print(f"Number of stored values: {csr_mat.nnz} / {2 * (m * n) * (m * n)}.")
