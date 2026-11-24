"""Unit tests for CSR sparse matrix creation and confidence weighting."""

import numpy as np
import pytest
from scipy import sparse


def test_csr_matrix_confidence_weighting():
    # 2 users, 3 items
    # user 0 played item 1 10 times -> c_01 = 1 + 40 * ln(1 + 10) = 1 + 40 * ln(11) = 1 + 40 * 2.3979 = 96.9158
    # user 1 played item 2 1 time -> c_12 = 1 + 40 * ln(2) = 1 + 40 * 0.6931 = 28.7259
    users = np.array([0, 1], dtype=np.int32)
    items = np.array([1, 2], dtype=np.int32)
    counts = np.array([10.0, 1.0], dtype=np.float32)
    
    alpha = 40.0
    confs = 1.0 + alpha * np.log1p(counts)
    
    mat = sparse.csr_matrix((confs, (users, items)), shape=(2, 3), dtype=np.float32)
    
    assert mat.shape == (2, 3)
    assert mat.nnz == 2
    assert pytest.approx(mat[0, 1]) == 1.0 + 40.0 * np.log(11.0)
    assert pytest.approx(mat[1, 2]) == 1.0 + 40.0 * np.log(2.0)
    assert mat[0, 0] == 0.0


def test_index_mapping_contiguity():
    users = ["user_c", "user_a", "user_b"]
    sorted_users = sorted(users)
    user_to_idx = {u: i for i, u in enumerate(sorted_users)}
    
    # Must be 0-indexed contiguous
    assert user_to_idx["user_a"] == 0
    assert user_to_idx["user_b"] == 1
    assert user_to_idx["user_c"] == 2
    assert set(user_to_idx.values()) == {0, 1, 2}
