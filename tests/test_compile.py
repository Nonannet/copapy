import pytest

from runner_helpers import NATIVE_RUNNER, run_vector_test


@pytest.mark.runner
def test_compile() -> None:
    run_vector_test('native', NATIVE_RUNNER)


if __name__ == "__main__":
    test_compile()
