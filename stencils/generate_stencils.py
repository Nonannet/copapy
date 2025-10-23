from typing import Generator, Callable
import argparse
from pathlib import Path
import os

op_signs = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**',
            'gt': '>', 'eq': '==', 'ne': '!=', 'mod': '%'}

entry_func_prefix = ''
stencil_func_prefix = '__attribute__((naked)) '  # Remove callee prolog

stack_size = 64

includes = ['aux_functions.c']


def read_files(files: list[str]) -> str:
    ret = ''
    script_dir = Path(__file__).parent
    for file_name in files:
        file_path = script_dir / file_name
        if not os.path.exists(file_path):
            file_path = Path(file_name)
        with open(file_path) as f:
            ret += f.read().strip(' \n') + '\n\n'
    return ret


def normalize_indent(text: str) -> str:
    text_lines = text.splitlines()
    if len(text_lines) > 1 and not text_lines[0].strip():
        text_lines = text_lines[1:]

    if not text_lines:
        return ''

    if len(text_lines) > 1 and text_lines[0] and text_lines[0][0] != ' ':
        indent_amount = len(text_lines[1]) - len(text_lines[1].lstrip())
    else:
        indent_amount = len(text_lines[0]) - len(text_lines[0].lstrip())

    return '\n' + '\n'.join(
        [' ' * max(0, len(line) - len(line.strip()) - indent_amount) + line.strip()
         for line in text_lines])


def norm_indent(f: Callable[..., str]) -> Callable[..., str]:
    return lambda *x: normalize_indent(f(*x))


@norm_indent
def get_entry_function_shell() -> str:
    return f"""
    {entry_func_prefix}int entry_function_shell(){{
        volatile char stack_place_holder[{stack_size}];
        stack_place_holder[0] = 0;
        result_int(0);
        return 1;
    }}
    """


@norm_indent
def get_op_code(op: str, type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(arg1 {op_signs[op]} arg2, arg2);
    }}
    """


@norm_indent
def get_cast(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void cast_{type_out}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(({type1})arg1, arg2);
    }}
    """


@norm_indent
def get_conv_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void conv_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(({type_out})arg1, arg2);
    }}
    """


@norm_indent
def get_op_code_float(op: str, type1: str, type2: str) -> str:
    return f"""
    {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}((float)arg1 {op_signs[op]} (float)arg2, arg2);
    }}
    """


@norm_indent
def get_pow(type1: str, type2: str) -> str:
    return f"""
    {stencil_func_prefix}void pow_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        //result_float_{type2}((float)math_pow((double)arg1, (double)arg2), arg2);
        result_float_{type2}(fast_pow_float((float)arg1, (float)arg2), arg2);
    }}
    """


@norm_indent
def get_floordiv(op: str, type1: str, type2: str) -> str:
    if type1 == 'int' and type2 == 'int':
        return f"""
        {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
            result_int_{type2}(floor_div((float)arg1, (float)arg2), arg2);
        }}
        """
    else:
        return f"""
        {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
            result_float_{type2}((float)floor_div((float)arg1, (float)arg2), arg2);
        }}
        """


@norm_indent
def get_result_stubs1(type1: str) -> str:
    return f"""
    void result_{type1}({type1} arg1);
    """


@norm_indent
def get_result_stubs2(type1: str, type2: str) -> str:
    return f"""
    void result_{type1}_{type2}({type1} arg1, {type2} arg2);
    """


@norm_indent
def get_read_reg0_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void read_{type_out}_reg0_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(dummy_{type_out}, arg2);
    }}
    """


@norm_indent
def get_read_reg1_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void read_{type_out}_reg1_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type1}_{type_out}(arg1, dummy_{type_out});
    }}
    """


@norm_indent
def get_write_code(type1: str) -> str:
    return f"""
    {stencil_func_prefix}void write_{type1}({type1} arg1) {{
        dummy_{type1} = arg1;
        result_{type1}(arg1);
    }}
    """


def permutate(*lists: list[str]) -> Generator[list[str], None, None]:
    if len(lists) == 0:
        yield []
        return
    first, *rest = lists
    for item in first:
        for items in permutate(*rest):
            yield [item, *items]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Output file path")
    parser.add_argument("--abi", type=str, default="", help="Optionaler String (Standard: '')")
    args = parser.parse_args()

    if args.abi:
        entry_func_prefix = f"__attribute__(({args.abi}_abi)) "

    code = "// Auto-generated stencils for copapy - Do not edit manually\n\n"

    code += read_files(includes)

    # Scalar arithmetic:
    types = ['int', 'float']
    ops = ['add', 'sub', 'mul', 'div', 'floordiv', 'gt', 'eq', 'ne', 'pow']

    for t1 in types:
        code += get_result_stubs1(t1)

    for t1, t2 in permutate(types, types):
        code += get_result_stubs2(t1, t2)

    code += get_entry_function_shell()

    for t1, t2 in permutate(types, types):
        t_out = 'int' if t1 == 'float' else 'float'
        code += get_cast(t1, t2, t_out)

    for op, t1, t2 in permutate(ops, types, types):
        t_out = t1 if t1 == t2 else 'float'
        if op == 'floordiv':
            code += get_floordiv('floordiv', t1, t2)
        elif op == 'div':
            code += get_op_code_float(op, t1, t2)
        elif op == 'pow':
            code += get_pow(t1, t2)
        elif op == 'gt' or op == 'eq' or op == 'ne':
            code += get_op_code(op, t1, t2, 'int')
        else:
            code += get_op_code(op, t1, t2, t_out)

    code += get_op_code('mod', 'int', 'int', 'int')

    for t1, t2, t_out in permutate(types, types, types):
        code += get_read_reg0_code(t1, t2, t_out)
        code += get_read_reg1_code(t1, t2, t_out)

    for t1 in types:
        code += get_write_code(t1)

    print(f"Write file {args.path}...")
    with open(args.path, 'w') as f:
        f.write(code)
