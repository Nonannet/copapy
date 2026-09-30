import pytest

import copapy as cp
from copapy import value
from runner_helpers import NATIVE_RUNNER, run_program, write_program


@pytest.mark.runner
def test_compile() -> None:
    test_vals = [0.0, -1.5, -2.0, -2.5, -3.0]

    # Function with no passing-on-jump as last instruction:
    ret = [cp.tan(value(v)) for v in test_vals]

    write_program(ret, cp.generic_sdb, 'build/runner/test.copapy')
    run_program(NATIVE_RUNNER, 'build/runner/test.copapy')


if __name__ == "__main__":
    test_compile()
