import pytest

import copapy as cp
from copapy import value
from runner_helpers import NATIVE_RUNNER, run_program, write_program


@pytest.mark.runner
def test_compile() -> None:
    ret = [value(16) / 2]

    write_program(ret, cp.generic_sdb, 'build/runner/test.copapy')
    run_program(NATIVE_RUNNER, 'build/runner/test.copapy')


if __name__ == "__main__":
    test_compile()
