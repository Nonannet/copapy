"""The tensor and matrix tests with all tensors packed into arrays
(array stencils for every operation that supports them)"""
from typing import Generator

import pytest

import copapy as cp
from test_matrix import *  # noqa: F401,F403
from test_tensor_basic import *  # noqa: F401,F403

# Constant matrices with more than one nonzero element are packed with a threshold
# of 1, the elimination of multiplications by 0 and 1 is tested in test_matrix only
del test_constant_matrices_are_not_variables  # noqa: F821


@pytest.fixture(autouse=True)
def pack_all_tensors() -> Generator[None, None, None]:
    threshold = cp.tensor.pack_threshold
    cp.tensor.pack_threshold = 1
    yield
    cp.tensor.pack_threshold = threshold
