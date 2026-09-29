from typing import Callable

import pytest

import copapy as cp
from copapy import value
from runner_helpers import NATIVE_RUNNER, run_program, write_program

test_vals = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0]


@pytest.mark.runner
@pytest.mark.parametrize('func', [cp.sqrt, cp.log, cp.sin])
def test_compile(func: Callable[[value[float]], value[float]]) -> None:
    ret = [func(value(v)) for v in test_vals]

    write_program(ret, cp.generic_sdb, 'build/runner/test.copapy')
    run_program(NATIVE_RUNNER, 'build/runner/test.copapy')


if __name__ == "__main__":
    for f in (cp.sqrt, cp.log, cp.sin):
        test_compile(f)
