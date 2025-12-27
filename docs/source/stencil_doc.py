import re
from pathlib import Path
import glob
import argparse

def extract_c_functions(stencils_path: str) -> dict[str, str]:
    """
    Extract all C function names and their code from a stencils.c file.

    Args:
        stencils_path: Path to the stencils.c file

    Returns:
        Dictionary mapping function names to their complete code
    """
    with open(stencils_path, 'r') as f:
        content = f.read()

    # Regex pattern to match C functions
    # Matches: return_type function_name(parameters) { ... }
    pattern = r'((?:STENCIL\s+extern|void|int|float|double)\s+\w+\s*\([^)]*\)\s*\{(?:[^{}]|\{[^{}]*\})*\})'

    functions: dict[str, str] = {}

    # Find all function matches
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        func_code = match.group(1).strip()

        # Extract function name using a simpler regex on the matched code
        name_match = re.search(r'(?:STENCIL\s+extern)?\s*(?:void|int|float|double)?\s*(\w+)\s*\(', func_code)

        if name_match:
            func_name = name_match.group(1)
            functions[func_name] = func_code

    return functions


def extract_asm_section(asm_path: str) -> dict[str, str]:
    """
    Extract assembly functions organized by section.

    Args:
        asm_path: Path to the stencils.asm file

    Returns:
        Dictionary with sections as keys, containing function dictionaries
    """
    with open(asm_path, 'r') as f:
        content = f.read()

    # Split by "Disassembly of section"
    sections = re.split(r'^Disassembly of section (.+?):', content, flags=re.MULTILINE)

    result: dict[str, str] = {}

    # Process sections (skip first empty element)
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        section_content = sections[i + 1] if i + 1 < len(sections) else ""

        if section_content:
            result[section_name] = section_content.strip()

    return result


def build_asm_code_dict(asm_glob_pattern: str) -> dict[str, dict[str, str]]:
    """
    Build a dictionary of assembly code for all available architectures.

    Args:
        asm_glob_pattern: Glob pattern to find stencils.asm files

    Returns:
        Dictionary mapping architecture names to their asm_code dictionaries
    """
    asm_code: dict[str, dict[str, str]] = {}

    asm_files = glob.glob(asm_glob_pattern)

    for asm_file in asm_files:
        arch_name = Path(asm_file).parent.name.replace('runner-linux-', '')

        try:
            asm_code[arch_name] = extract_asm_section(asm_file)
            print(f"Loaded assembly for {arch_name}")
        except FileNotFoundError:
            print(f"Warning: Assembly file not found for {arch_name}: {asm_file}")

    return asm_code


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stencils documentation from C and assembly code")
    parser.add_argument('--input', default='build/stencils.c', help='Path to input C file')
    parser.add_argument('--asm-pattern', default='build/tmp/runner-*/stencils.asm', help='Glob pattern for assembly files')
    parser.add_argument('--output', default='docs/build/stencils.md', help='Output markdown file path')

    args = parser.parse_args()

    # Get all C functions
    functions = extract_c_functions(args.input)

    # Build assembly code dictionary for all architectures
    asm_code = build_asm_code_dict(args.asm_pattern)

    #@norm_indent
    def get_stencil_section(func_name: str) -> str:
        c_code = functions[func_name]
        section_name = '.text.' + func_name

        arch_asm_code = ''
        for arch in sorted(asm_code.keys()):
            if section_name in asm_code[arch]:
                arch_asm_code += f"""
### {arch}
```nasm
{asm_code[arch][section_name]}
```
"""
            else:
                arch_asm_code += f"\n### {arch}\nNo assembly found for this architecture\n"

        return f"""
## {func_name}
```c
{c_code}
```
{arch_asm_code}
"""

    md_code: str = ''

    for function_name, code in functions.items():
        md_code += get_stencil_section(function_name)

    with open(args.output, 'wt') as f:
        f.write(md_code)

    print(f"Generated {args.output} with {len(functions)} stencil functions")
