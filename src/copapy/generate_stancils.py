from typing import Generator


op_signs = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}

def get_op_code(op: str, type1: str, type2: str, type_out: str):
    return f"""
    void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        asm volatile (".long 0xF17ECAFE");
        result_{type_out}_{type2}(arg1 {op_signs[op]} arg2, arg2);
        asm volatile (".long 0xF27ECAFE");
    }}
    """

def get_result_stubs(type1: str, type2: str):
    return f"""
    void result_{type1}_{type2}({type1} arg1, {type2} arg2);
    """

def get_read_reg0_code(type1: str, type2: str, type_out: str):
    return f"""
    void read_reg0_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        asm volatile (".long 0xF17ECAFE");
        result_{type_out}_{type2}(dummy_{type_out}, arg2);
        asm volatile (".long 0xF27ECAFE");
    }}
    """

def get_read_reg1_code(type1: str, type2: str, type_out: str):
    return f"""
    void read_reg1_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        asm volatile (".long 0xF17ECAFE");
        result_{type1}_{type_out}(arg1, dummy_{type_out});
        asm volatile (".long 0xF27ECAFE");
    }}
    """

def get_write_code(type1: str, type2: str):
    return f"""
    void write_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        asm volatile (".long 0xF17ECAFE");
        dummy_{type1} = arg1;
        result_{type1}_{type2}(arg1, arg2);
        asm volatile (".long 0xF27ECAFE");
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
    types = ['int', 'float']
    ops = ['add', 'sub', 'mul', 'div']

    code = """
    // Auto-generated stancils for copapy
    // Do not edit manually

    volatile int dummy_int = 1337;
    volatile float dummy_float = 1337;
    """
    for t1, t2 in permutate(types, types):
        code += get_result_stubs(t1, t2)

    for op, t1, t2 in permutate(ops, types, types):
        t_out = t1 if t1 == t2 else 'float'
        code += get_op_code(op, t1, t2, t_out)

    for t1, t2, t_out in permutate(types, types, types):
        code += get_read_reg0_code(t1, t2, t_out)
        code += get_read_reg1_code(t1, t2, t_out)
    
    for t1, t2 in permutate(types, types):
        code += get_write_code(t1, t2)

    with open('src/copapy/stancils.c', 'w') as f:
        f.write(code)
