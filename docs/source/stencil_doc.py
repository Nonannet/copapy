import re
from pathlib import Path
import glob
import argparse
from docutils.nodes import make_id
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

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

    # Function header at the start of a line: [qualifiers] return_type name(parameters) {
    header_pattern = r'^[A-Za-z_][^\n;{}#=]*?\b(\w+)\s*\([^;{}]*\)\s*\{'

    functions: dict[str, str] = {}

    for match in re.finditer(header_pattern, content, re.MULTILINE):
        end = find_block_end(content, match.end())
        if end > 0:
            functions[match.group(1)] = content[match.start():end]

    return functions


def find_block_end(code: str, pos: int) -> int:
    """
    Find the end of a C block by counting braces, ignoring braces in
    comments, string and character literals.

    Args:
        code: C source code
        pos: Position directly after the opening brace of the block

    Returns:
        Position after the closing brace, -1 if the block is not closed
    """
    token_pattern = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.DOTALL)
    depth = 1
    for token in token_pattern.finditer(code, pos):
        if token.group() == '{':
            depth += 1
        elif token.group() == '}':
            depth -= 1
            if depth == 0:
                return token.end()
    return -1


def extract_stencil_names(stencils_path: str) -> set[str]:
    """
    Extract the names of all functions declared with the STENCIL macro.

    Args:
        stencils_path: Path to the stencils.c file

    Returns:
        Set of stencil function names
    """
    with open(stencils_path, 'r') as f:
        content = f.read()

    return set(re.findall(r'^.*\bSTENCIL\b[^(\n]*?(\w+)\s*\(', content, re.MULTILINE))


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


# Category key and page title; each key needs a matching stencil_doc_<key>.md page
CATEGORIES: list[tuple[str, str]] = [
    ('load_store', 'Load and store stencils'),
    ('conversion', 'Conversion and copy stencils'),
    ('scalar_arith', 'Scalar arithmetic and logic stencils'),
    ('scalar_math', 'Scalar math stencils'),
    ('array_elementwise', 'Array element-wise stencils'),
    ('array_linalg', 'Array reduction and linear algebra stencils'),
    ('aux', 'Auxiliary functions'),
]

LOAD_STORE_OPS = {'load', 'store'}
CONVERSION_OPS = {'float', 'int', 'bool', 'byte', 'copy32', 'copy8', 'roll32', 'put32'}
SCALAR_MATH_OPS = {'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                   'atan2', 'tanh', 'pow'}
ARRAY_LINALG_OPS = {'dot', 'matvec', 'matmul', 'solve', 'det', 'lerp', 'bsearch', 'conv2d',
                    'maxpool2d', 'avgpool2d', 'sum', 'sumaxis', 'sort', 'argsort'}
ARRAY_REDUCTION_OPS = {'min', 'max'}  # Reductions if called with a single array argument


def get_category(func_name: str, stencil_names: set[str]) -> str:
    """
    Determine the documentation category of a function by its name.

    Args:
        func_name: Function name, e.g. "add_floatarr_float"
        stencil_names: Names of all STENCIL functions; others are auxiliary functions

    Returns:
        Category key from CATEGORIES
    """
    if func_name not in stencil_names:
        return 'aux'

    op, *arg_types = func_name.split('_')
    is_array = any(t.endswith('arr') for t in arg_types)

    if op in LOAD_STORE_OPS:
        return 'load_store'
    if op in CONVERSION_OPS:
        return 'conversion'
    if is_array:
        if op in ARRAY_LINALG_OPS or (op in ARRAY_REDUCTION_OPS and len(arg_types) == 1):
            return 'array_linalg'
        return 'array_elementwise'
    if op in SCALAR_MATH_OPS:
        return 'scalar_math'
    return 'scalar_arith'


def highlight_code(code: str, language: str, link_names: list[str]) -> str:
    """
    Syntax highlight code as a raw HTML block in which all references to the
    given auxiliary functions link to their entry on the auxiliary functions page.

    Args:
        code: Source or disassembly code
        language: Pygments lexer name, e.g. "c" or "objdump"
        link_names: Names of the auxiliary functions to link

    Returns:
        MyST raw HTML block with the same markup Sphinx uses for code blocks
    """
    html_code = highlight(code, get_lexer_by_name(language), HtmlFormatter(nowrap=True))

    if link_names:
        # Longest names first, so that e.g. aux_arr_sum_int is not matched as aux_arr_sum
        names = sorted(link_names, key=len, reverse=True)
        pattern = r'(?<!\w)(' + '|'.join(re.escape(n) for n in names) + r')(?!\w)'
        html_code = re.sub(
            pattern,
            lambda m: f'<a class="aux-call" href="stencil_doc_aux.html#{make_id(m.group(1))}">{m.group(1)}</a>',
            html_code)

    return (f'````{{raw}} html\n<div class="highlight-{language} notranslate"><div class="highlight">'
            f'<pre>{html_code}</pre></div></div>\n````')


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stencils documentation from C and assembly code")
    parser.add_argument('--input', default='build/stencils.c', help='Path to input C file')
    parser.add_argument('--asm-pattern', default='build/tmp/runner-*/stencils.asm', help='Glob pattern for assembly files')
    parser.add_argument('--output', default='docs/build/stencils.md', help='Output markdown file path')

    args = parser.parse_args()

    # Get all C functions
    functions = extract_c_functions(args.input)
    stencil_names = extract_stencil_names(args.input)

    # Build assembly code dictionary for all architectures
    asm_code = build_asm_code_dict(args.asm_pattern)

    documented = [name for name in functions if 'get_42' not in name]
    aux_names = [name for name in documented if get_category(name, stencil_names) == 'aux']

    #@norm_indent
    def get_stencil_section(func_name: str) -> str:
        c_code = functions[func_name]
        section_name = '.text.' + func_name
        link_names = [name for name in aux_names if name != func_name]

        arch_asm_code = ''
        for arch in sorted(asm_code.keys()):
            if section_name in asm_code[arch]:
                arch_asm_code += f"""
### {arch}
{highlight_code(asm_code[arch][section_name], 'objdump', link_names)}
"""
            else:
                arch_asm_code += f"\n### {arch}\nNo assembly found for this architecture\n"

        return f"""
## {func_name}
{highlight_code(c_code, 'c', link_names)}
{arch_asm_code}
"""

    category_md: dict[str, str] = {key: '' for key, _ in CATEGORIES}

    for function_name in documented:
        category_md[get_category(function_name, stencil_names)] += get_stencil_section(function_name)

    # One file per category: <output stem>_<category key><output suffix>
    output_path = Path(args.output)
    for key, title in CATEGORIES:
        category_path = output_path.with_name(f"{output_path.stem}_{key}{output_path.suffix}")
        with open(category_path, 'wt') as f:
            f.write(category_md[key])
        print(f"Generated {category_path} ({title})")

    print(f"Generated documentation for {len(functions)} stencil functions")
