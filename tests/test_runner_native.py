import pytest

import copapy as cp
from runner_helpers import NATIVE_RUNNER, TEST_PROGRAMS, run_runner_test, trig_tol

ARCH = 'native'


@pytest.mark.runner
def test_ops() -> None:
    run_runner_test('ops', ARCH, NATIVE_RUNNER)


@pytest.mark.runner
def test_math() -> None:
    run_runner_test('math', ARCH, NATIVE_RUNNER)


@pytest.mark.runner
def test_vector() -> None:
    run_runner_test('vector', ARCH, NATIVE_RUNNER)


@pytest.mark.runner
def test_array() -> None:
    run_runner_test('array', ARCH, NATIVE_RUNNER)


@pytest.mark.parametrize('name', list(TEST_PROGRAMS))
def test_target(name: str) -> None:
    """Same test programs executed with the coparun Python module"""
    ret_test, ret_ref = TEST_PROGRAMS[name]()

    tg = cp.Target()
    tg.compile(ret_test)
    tg.run()

    for test, ref in zip(ret_test, ret_ref):
        assert isinstance(test, cp.value)
        val = tg.read_value(test)
        print('+', val, ref, test.dtype)
        for t in (int, float, bool):
            assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, trig_tol(), trig_tol()), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_ops()
    test_math()
    test_vector()
    test_array()
    for n in TEST_PROGRAMS:
        test_target(n)
