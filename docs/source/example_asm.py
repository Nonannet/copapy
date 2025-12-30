from pathlib import Path
import glob
import argparse


def build_asm_code_dict(asm_glob_pattern: str) -> dict[str, str]:
    """
    Build a dictionary of assembly code for all available architectures.

    Args:
        asm_glob_pattern: Glob pattern to find stencils.asm files

    Returns:
        Dictionary mapping architecture names to their asm_code dictionaries
    """
    asm_code: dict[str, str] = {}

    # Find all stencils.asm files matching the pattern
    asm_files = glob.glob(asm_glob_pattern)

    for asm_file in asm_files:
        arch_name = Path(asm_file).parent.name.replace('runner-linux-', '')

        try:
            with open(asm_file) as f:
                asm_code[arch_name] = f.read()
            print(f"Loaded assembly for {arch_name}")
        except FileNotFoundError:
            print(f"Warning: Assembly file not found for {arch_name}: {asm_file}")

    return asm_code


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stencils documentation from C and assembly code")
    parser.add_argument('--input', default='tools/make_example.py', help='Path to example script')
    parser.add_argument('--asm-pattern', default='build/tmp/runner-linux-*/example.asm', help='Glob pattern for assembly files')
    parser.add_argument('--output', default='docs/build/compiled_example.md', help='Output markdown file path')

    args = parser.parse_args()

    # Build assembly code dictionary for all architectures
    asm_code = build_asm_code_dict(args.asm_pattern)

    with open(args.input) as f:
        python_code = f.read()

    md_code: str = f"""
Example program:
```python
{python_code}
```
"""

    for arch in sorted(asm_code.keys()):
        md_code += f"""
## {arch}
```nasm
{asm_code[arch]}
```
"""

    with open(args.output, 'wt') as f:
        f.write(md_code)

    print(f"Generated {args.output} for {len(asm_code)} architectures")
