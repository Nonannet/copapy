#!/usr/bin/env python3
"""
Load two raw blobs into qemu-system-tricore memory and execute them
"""

import argparse
import subprocess
import sys
import shutil
import os
import tempfile
from pathlib import Path

if os.name == 'nt':
    QEMU_BIN = ['wsl', '/opt/tricore/bin/qemu-system-tricore']
else:
    QEMU_BIN = ['qemu-system-tricore']

MACHINE = "tc27x"
RESET_VECTOR = 0x80000000

def check_qemu():
    if os.name == 'nt':
        # On Windows, just check if wsl is available
        if shutil.which('wsl') is None:
            print("Cannot find 'wsl' in PATH")
            sys.exit(1)
    else:
        if shutil.which(QEMU_BIN[0]) is None:
            print(f"Cannot find {QEMU_BIN[0]} in PATH")
            sys.exit(1)

def make_jump_trampoline(entry_addr: int, out_file: Path):
    """
    Generate a TriCore jump instruction to entry_addr from RESET_VECTOR.
    Uses a 24-bit relative jump instruction: j offset
    
    Instruction encoding:
    - Byte 0: 0x1d (opcode for relative jump)
    - Bytes 1-3: 24-bit signed relative offset
    
    Raises an error if the offset doesn't fit in 24 bits.
    """
    # Calculate relative offset from RESET_VECTOR to entry_addr
    # offset = entry_addr - (RESET_VECTOR + instruction_size)
    offset = entry_addr - (RESET_VECTOR + 4)
    
    # Check if offset fits in 24-bit signed range: [-2^23, 2^23-1]
    max_24bit = (1 << 23) - 1
    min_24bit = -(1 << 23)
    
    if offset < min_24bit or offset > max_24bit:
        raise ValueError(
            f"24-bit relative jump is not sufficient. "
            f"Offset {offset} (0x{offset:x}) is outside [-2^23, 2^23-1] range. "
            f"RESET_VECTOR: 0x{RESET_VECTOR:x}, entry_addr: 0x{entry_addr:x}"
        )

    # Encode the 24-bit offset in little-endian format (3 bytes)
    offset_bytes = offset.to_bytes(3, byteorder='little', signed=True)

    # Create the jump instruction: opcode (0x1d) + 24-bit offset
    instruction = bytes([0x1d]) + offset_bytes

    out_file.write_bytes(instruction)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="code blob")
    ap.add_argument("--data", required=True, help="data blob")
    ap.add_argument("--code-addr", required=True, type=lambda x:int(x,0))
    ap.add_argument("--data-addr", required=True, type=lambda x:int(x,0))
    ap.add_argument("--entry", required=True, type=lambda x:int(x,0))
    ap.add_argument("--machine", default=MACHINE)
    args = ap.parse_args()

    check_qemu()

    # Create temporary file for trampoline
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tramp = Path(tmp.name)
    
    try:
        make_jump_trampoline(args.entry, tramp)

        cmd = list(QEMU_BIN) + [
            "-M", args.machine,
            "-nographic",

            # reset vector trampoline
            "-device", f"loader,file={tramp},addr=0x{RESET_VECTOR:x}",

            # user code/data
            "-device", f"loader,file={args.code},addr=0x{args.code_addr:x}",
            "-device", f"loader,file={args.data},addr=0x{args.data_addr:x}",

            # logs
            "-d", "in_asm,exec,cpu,guest_errors",
            "-D", "qemu.log",
        ]

        print("Running:")
        print(" ".join(cmd))
        print()

        subprocess.run(cmd)
    finally:
        # Clean up temporary file
        if tramp.exists():
            tramp.unlink()

if __name__ == "__main__":
    main()