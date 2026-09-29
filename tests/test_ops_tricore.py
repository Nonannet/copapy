import os

import pytest

from runner_helpers import run_ops_test, run_vector_test

# No qemu-user exists for TriCore: The bare metal runner is run in
# qemu-system-tricore by this script (on Windows in WSL)
QEMU_SCRIPT = (['wsl'] if os.name == 'nt' else []) + ['sh', 'tools/tricore/run_qemu.sh']
RUNNER = 'build/runner/coparun-tricore.elf'


@pytest.mark.runner
def test_compile() -> None:
    run_ops_test('tricore', RUNNER, QEMU_SCRIPT)


@pytest.mark.runner
def test_vector() -> None:
    run_vector_test('tricore', RUNNER, QEMU_SCRIPT)


if __name__ == "__main__":
    test_compile()
    test_vector()
