#!/bin/sh
# Runs a copapy command file with the bare metal runner (src/coparun/coparun_baremetal.c)
# in qemu-system-tricore. Since there is no qemu-user mode for TriCore, the command file
# is loaded to the runner's command buffer and the runner output is read back
# by tricore-elf-gdb from the memory, when the runner reaches coparun_done.
#
# Usage: run_qemu.sh <runner.elf> <code_file> [memory_dump_file]
#        run_qemu.sh --version
#
# Prints the runner output and exits with the runner's return code.
# Requires qemu-system-tricore and tricore-elf-gdb in PATH or in /opt/tricore/bin.

set -eu

PATH="$PATH:/opt/tricore/bin"
QEMU=qemu-system-tricore
GDB=tricore-elf-gdb
COMMAND_BUFFER_ADDR=0xa1300000  # must match coparun_baremetal.c
TIMEOUT=${COPARUN_TIMEOUT:-60}

if [ "${1:-}" = "--version" ]; then
    command -v $GDB > /dev/null || { echo "$GDB not found" >&2; exit 1; }
    exec $QEMU --version
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <runner.elf> <code_file> [memory_dump_file]" >&2
    exit 1
fi

RUNNER=$1
CODE_FILE=$2
DUMP_FILE=${3:-}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

DUMP_CMD=""
if [ -n "$DUMP_FILE" ]; then
    DUMP_CMD="if targ.executable_memory_len > 0
dump binary memory $DUMP_FILE targ.executable_memory targ.executable_memory+targ.executable_memory_len
end"
fi

# qemu is started by gdb and communicates with it over stdio (-gdb stdio),
# the runner stops at startup (-S) until gdb continues
cat > "$TMP_DIR/cmds.gdb" <<EOF
set pagination off
set confirm off
target remote | exec $QEMU -M tricore_testboard -cpu tc27x -display none -monitor none -serial none -S -gdb stdio -kernel $RUNNER -device loader,file=$CODE_FILE,addr=$COMMAND_BUFFER_ADDR
break coparun_done
continue
if coparun_output_len > 0
dump binary memory $TMP_DIR/output.txt coparun_output coparun_output+coparun_output_len
end
$DUMP_CMD
printf "COPARUN_RETURN=%d\n", ret
quit
EOF

GDB_RET=0
timeout "$TIMEOUT" $GDB -batch -nx -x "$TMP_DIR/cmds.gdb" "$RUNNER" > "$TMP_DIR/gdb.txt" 2>&1 || GDB_RET=$?
if [ $GDB_RET -eq 124 ]; then
    echo "Running $RUNNER in $QEMU timed out ($TIMEOUT s):" >&2
    cat "$TMP_DIR/gdb.txt" >&2
    exit 124
fi

RET=$(sed -n 's/^COPARUN_RETURN=\([0-9-]*\).*/\1/p' "$TMP_DIR/gdb.txt")
if [ -z "$RET" ]; then
    echo "Runner did not reach coparun_done:" >&2
    cat "$TMP_DIR/gdb.txt" >&2
    exit 125
fi

[ -f "$TMP_DIR/output.txt" ] && cat "$TMP_DIR/output.txt"
exit "$RET"
