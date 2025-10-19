from typing import Generator
import argparse

op_signs = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**',
            'gt': '>', 'eq': '==', 'ne': '!=', 'mod': '%'}

entry_func_prefix = ''
stencil_func_prefix = '__attribute__((naked)) '  # Remove callee prolog

stack_size = 64


def get_aux_funcs() -> str:
    return f"""
    {entry_func_prefix}int entry_function_shell(){{
        volatile char stack_place_holder[{stack_size}];
        stack_place_holder[0] = 0;
        result_int(0);
        return 1;
    }}

    """ + """
    __attribute__((noinline)) int floor_div(float arg1, float arg2) {
        float x = arg1 / arg2;
        int i = (int)x;
        if (x < 0 && x != (float)i) i -= 1;
        return i;
    }
    """


def get_op_code(op: str, type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(arg1 {op_signs[op]} arg2, arg2);
    }}
    """


def get_cast(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void cast_{type_out}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(({type1})arg1, arg2);
    }}
    """


def get_conv_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void conv_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(({type_out})arg1, arg2);
    }}
    """


def get_op_code_float(op: str, type1: str, type2: str) -> str:
    return f"""
    {stencil_func_prefix}void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}((float)arg1 {op_signs[op]} (float)arg2, arg2);
    }}
    """


def get_pow(type1: str, type2: str) -> str:
    return f"""
    {stencil_func_prefix}void pow_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}((float)math_pow((double)arg1, (double)arg2), arg2);
    }}
    """


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


def get_result_stubs1(type1: str) -> str:
    return f"""
    void result_{type1}({type1} arg1);
    """


def get_result_stubs2(type1: str, type2: str) -> str:
    return f"""
    void result_{type1}_{type2}({type1} arg1, {type2} arg2);
    """


def get_read_reg0_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void read_{type_out}_reg0_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(dummy_{type_out}, arg2);
    }}
    """


def get_read_reg1_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    {stencil_func_prefix}void read_{type_out}_reg1_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type1}_{type_out}(arg1, dummy_{type_out});
    }}
    """


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

    code = """
    // Auto-generated stencils for copapy
    // Do not edit manually

    double (*math_pow)(double, double);

    volatile int dummy_int = 1337;
    volatile float dummy_float = 1337;
    """

    # Scalar arithmetic:
    types = ['int', 'float']
    ops = ['add', 'sub', 'mul', 'div', 'floordiv', 'gt', 'eq', 'ne', 'pow']

    for t1 in types:
        code += get_result_stubs1(t1)

    for t1, t2 in permutate(types, types):
        code += get_result_stubs2(t1, t2)

    code += get_aux_funcs()

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

    with open(args.path, 'w') as f:
        f.write(code)
