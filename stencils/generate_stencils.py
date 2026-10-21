from typing import Generator, Callable
import argparse
from pathlib import Path
import os

op_signs = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**',
            'gt': '>', 'eq': '==', 'ge': '>=', 'ne': '!=', 'mod': '%',
            'lshift': '<<', 'rshift': '>>',
            'bwand': '&', 'bwor': '|', 'bwxor': '^'}

entry_func_prefix = ''

stack_size = 128

includes = ['stencil_helper.h', 'aux_functions.c']


def read_files(files: list[str]) -> str:
    ret: str = ''
    script_dir = Path(__file__).parent
    for file_name in files:
        file_path = script_dir / file_name
        if not os.path.exists(file_path):
            file_path = Path(file_name)
        with open(file_path) as f:
            ret += f.read().strip(' \n') + '\n\n'

    for incl in includes:
        ret = ret.replace(f'#include "{incl}"\n', '')

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
        //volatile char stack_place_holder[{stack_size}];
        //stack_place_holder[0] = 0;
        result_int(0);
        return 1;
    }}
    """


@norm_indent
def get_op_code(op: str, type1: str, type2: str, type_out: str) -> str:
    return f"""
    STENCIL void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(arg1 {op_signs[op]} arg2, arg2);
    }}
    """


@norm_indent
def get_cast(type_out: str, type_in: str) -> str:
    return f"""
    STENCIL void {type_out}_{type_in}({type_in} arg1) {{
        // Conversion of {type_in} to {type_out}
        result_{type_out}(({type_out})arg1);
    }}
    """


@norm_indent
def get_neg(type1: str) -> str:
    return f"""
    STENCIL void neg_{type1}({type1} arg1) {{
        result_{type1}(-arg1);
    }}
    """


@norm_indent
def get_func1(func_name: str, type1: str) -> str:
    return f"""
    STENCIL void {func_name}_{type1}({type1} arg1) {{
        result_float(aux_{func_name}((float)arg1));
    }}
    """


@norm_indent
def get_custom_stencil(stencil_signature: str, stencil_body: str, comment: str = '') -> str:
    if comment:
        stencil_body = f"// {comment}\n        {stencil_body}"
    return f"""
    STENCIL void {stencil_signature} {{
        {stencil_body}
    }}
    """


@norm_indent
def get_func2(func_name: str, type1: str, type2: str) -> str:
    return f"""
    STENCIL void {func_name}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}(aux_{func_name}((float)arg1, (float)arg2), arg2);
    }}
    """


@norm_indent
def get_math_func1(func_name: str, type1: str, stencil_name: str) -> str:
    return f"""
    STENCIL void {stencil_name}_{type1}({type1} arg1) {{
        result_float({func_name}((float)arg1));
    }}
    """


@norm_indent
def get_math_func2(func_name: str, type1: str, type2: str) -> str:
    return f"""
    STENCIL void {func_name}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}({func_name}f((float)arg1, (float)arg2), arg2);
    }}
    """


@norm_indent
def get_conv_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    STENCIL void conv_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(({type_out})arg1, arg2);
    }}
    """


@norm_indent
def get_op_code_float(op: str, type1: str, type2: str) -> str:
    return f"""
    STENCIL void {op}_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_float_{type2}((float)arg1 {op_signs[op]} (float)arg2, arg2);
    }}
    """


@norm_indent
def get_floordiv(op: str, type1: str, type2: str) -> str:
    if type1 == 'int' and type2 == 'int':
        return f"""
        STENCIL void {op}_{type1}_{type2}({type1} a, {type2} b) {{
            int result = a / b - ((a % b != 0) && ((a < 0) != (b < 0)));
            result_int_{type2}(result, b);
        }}
        """
    else:
        return f"""
        STENCIL void {op}_{type1}_{type2}({type1} a, {type2} b) {{
            result_float_{type2}(floorf((float)a / (float)b), b);
        }}
        """


@norm_indent
def get_min(type1: str, type2: str) -> str:
    if type1 == 'int' and type2 == 'int':
        return f"""
        STENCIL void min_{type1}_{type2}({type1} a, {type2} b) {{
            result_int_{type2}(a < b ? a : b, b);
        }}
        """
    else:
        return f"""
        STENCIL void min_{type1}_{type2}({type1} a, {type2} b) {{
            float _a = (float)a; float _b = (float)b;
            result_float_{type2}(_a < _b ? _a : _b, b);
        }}
        """


@norm_indent
def get_max(type1: str, type2: str) -> str:
    if type1 == 'int' and type2 == 'int':
        return f"""
        STENCIL void max_{type1}_{type2}({type1} a, {type2} b) {{
            result_int_{type2}(a > b ? a : b, b);
        }}
        """
    else:
        return f"""
        STENCIL void max_{type1}_{type2}({type1} a, {type2} b) {{
            float _a = (float)a; float _b = (float)b;
            result_float_{type2}(_a > _b ? _a : _b, b);
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
def get_load_reg0_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    STENCIL void load_{type_out}_reg0_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type_out}_{type2}(dummy_{type_out}, arg2);
    }}
    """


@norm_indent
def get_load_reg1_code(type1: str, type2: str, type_out: str) -> str:
    return f"""
    STENCIL void load_{type_out}_reg1_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        result_{type1}_{type_out}(arg1, dummy_{type_out});
    }}
    """


@norm_indent
def get_store_code(type1: str, type2: str) -> str:
    return f"""
    STENCIL void store_{type1}_reg0_{type1}_{type2}({type1} arg1, {type2} arg2) {{
        dummy_{type1} = arg1;
        result_{type1}_{type2}(arg1, arg2);
    }}
    """


def arr_operand_type(op: str, type1: str, type2: str) -> str:
    """Returns type of both operands of an array operation they must be casted to
    before the operation."""
    return 'float' if op in ('div', 'atan2') or type1 != type2 else type1


def arr_out_type(op: str, type1: str, type2: str) -> str:
    """Element type of the result of an array operation: int for comparisons,
    otherwise the type of the casted operands"""
    return 'int' if op in ('gt', 'ge', 'eq', 'ne') else arr_operand_type(op, type1, type2)


def arr_casted_operand(type1: str, x: str, type_out: str) -> str:
    """C expression x of type1 casted to type_out"""
    return x if type1 == type_out else f"({type_out}){x}"


@norm_indent
def get_arr_op_code(op: str, type1: str, type2: str, func: str = '') -> str:
    """Element-wise binary operation for the operand types type1 and type2:
    an operator (add, sub, ...), a comparison (gt, ge, eq, ne), min, max or a C function
    func(a, b)."""
    t_op = arr_operand_type(op, type1, type2)
    t_out = arr_out_type(op, type1, type2)
    kernel = f"aux_arr_{op}_{type1}_{type2}"

    def expr(x: str, y: str) -> str:
        """C expression for one element of the result, x and y are the C
        expressions of the first and second operand (element or scalar)"""
        x = arr_casted_operand(type1, x, t_op)
        y = arr_casted_operand(type2, y, t_op)
        if op == 'min':
            return f"({x} < {y} ? {x} : {y})"
        if op == 'max':
            return f"({x} > {y} ? {x} : {y})"
        if func:
            return f"{func}({x}, {y})"
        return f"{x} {op_signs[op]} {y}"

    return f"""
    KERNEL void {kernel}_vv(const {type1} *restrict a, const {type2} *restrict b, {t_out} *restrict o, int n) {{
        for (int i = 0; i < n; i++) o[i] = {expr('a[i]', 'b[i]')};
    }}

    KERNEL void {kernel}_vs(const {type1} *restrict a, {type2} b, {t_out} *restrict o, int n) {{
        for (int i = 0; i < n; i++) o[i] = {expr('a[i]', 'b')};
    }}

    KERNEL void {kernel}_sv({type1} a, const {type2} *restrict b, {t_out} *restrict o, int n) {{
        for (int i = 0; i < n; i++) o[i] = {expr('a', 'b[i]')};
    }}

    STENCIL void {op}_{type1}arr_{type2}arr(void) {{
        // Element-wise {op} of {type1} array and {type2} array
        {kernel}_vv(REF(ref_arg0), REF(ref_arg1), REF(ref_out), *(int *)REF(ref_arg2));
        result_{t_out}_ref();
    }}

    STENCIL void {op}_{type1}arr_{type2}(void) {{
        // Element-wise {op} of {type1} array and {type2} scalar
        {kernel}_vs(REF(ref_arg0), *({type2} *)REF(ref_arg1), REF(ref_out), *(int *)REF(ref_arg2));
        result_{t_out}_ref();
    }}

    STENCIL void {op}_{type1}_{type2}arr(void) {{
        // Element-wise {op} of {type1} scalar and {type2} array
        {kernel}_sv(*({type1} *)REF(ref_arg0), REF(ref_arg1), REF(ref_out), *(int *)REF(ref_arg2));
        result_{t_out}_ref();
    }}
    """


def get_reduction_body(type_out: str, term: Callable[[str], str], lanes: int = 8) -> str:
    """Function body returning the sum of term(i) for i in range(n)."""
    decl = ', '.join(f's{k} = 0' for k in range(lanes))
    body = ' '.join(f's{k} += {term(f"i + {k}")};' for k in range(lanes))
    total = ' + '.join(f's{k}' for k in range(lanes))
    return f"""// Independent partial sums allow vectorization without reassociation of floats.
        // Scalars instead of an accumulator array avoid memset calls for the initialization.
        {type_out} {decl}, r = 0;
        int i = 0;
        for (; i + {lanes} <= n; i += {lanes}) {{
            {body}
        }}
        for (; i < n; i++) r += {term('i')};
        return r + {total};"""


@norm_indent
def get_arr_dot_code(type1: str, type2: str) -> str:
    t_out = arr_out_type('mul', type1, type2)
    kernel = f"aux_arr_dot_{type1}_{type2}"
    body = get_reduction_body(t_out, lambda i: f"{arr_casted_operand(type1, f'a[{i}]', t_out)} * {arr_casted_operand(type2, f'b[{i}]', t_out)}")
    return f"""
    KERNEL {t_out} {kernel}(const {type1} *restrict a, const {type2} *restrict b, int n) {{
        {body}
    }}

    KERNEL void aux_arr_matvec_{type1}_{type2}(const {type1} *restrict a, const {type2} *restrict b, {t_out} *restrict o, int m, int n) {{
        for (int j = 0; j < m; j++) o[j] = {kernel}(a + j * n, b, n);
    }}

    STENCIL void dot_{type1}arr_{type2}arr(void) {{
        // Dot product of {type1} array and {type2} array
        *({t_out} *)REF(ref_out) = {kernel}(REF(ref_arg0), REF(ref_arg1), *(int *)REF(ref_arg2));
        result_{t_out}_ref();
    }}

    STENCIL void matvec_{type1}arr_{type2}arr(void) {{
        // Product of a row-major m x n matrix and a vector of n elements
        aux_arr_matvec_{type1}_{type2}(REF(ref_arg0), REF(ref_arg1), REF(ref_out), *(int *)REF(ref_arg2), *(int *)REF(ref_arg3));
        result_{t_out}_ref();
    }}
    """


@norm_indent
def get_arr_matmul_code(type1: str, type2: str) -> str:
    t_out = arr_out_type('mul', type1, type2)
    a = arr_casted_operand(type1, 'ar[q]', t_out)
    a0 = arr_casted_operand(type1, 'ar[0]', t_out)
    b = arr_casted_operand(type2, 'bq[j]', t_out)
    b0 = arr_casted_operand(type2, 'b[j]', t_out)
    return f"""
    KERNEL void aux_arr_matmul_{type1}_{type2}(const {type1} *restrict a, const {type2} *restrict b, {t_out} *restrict o, const int *restrict p) {{
        // The i-k-j loop order keeps the inner loop contiguous for vectorization,
        // the first k step initializes the output row
        int m = p[0], k = p[1], n = p[2];
        for (int i = 0; i < m; i++) {{
            {t_out} *restrict r = o + i * n;
            const {type1} *ar = a + i * k;
            {t_out} a0 = {a0};
            for (int j = 0; j < n; j++) r[j] = a0 * {b0};
            for (int q = 1; q < k; q++) {{
                {t_out} aq = {a};
                const {type2} *bq = b + q * n;
                for (int j = 0; j < n; j++) r[j] += aq * {b};
            }}
        }}
    }}

    STENCIL void matmul_{type1}arr_{type2}arr(void) {{
        // Matrix product of a row-major m x k and a k x n matrix,
        // the dimensions are passed as int array [m, k, n]
        aux_arr_matmul_{type1}_{type2}(REF(ref_arg0), REF(ref_arg1), REF(ref_out), REF(ref_arg2));
        result_{t_out}_ref();
    }}
    """


@norm_indent
def get_arr_solve_code() -> str:
    """The kernels aux_arr_solve_float and aux_arr_det_float are in aux_functions.c"""
    return """
    STENCIL void solve_floatarr_floatarr(void) {
        // Solution of a linear system
        aux_arr_solve_float(REF(ref_arg0), REF(ref_arg1), REF(ref_out), REF(ref_arg3), REF(ref_arg2));
        result_float_ref();
    }

    STENCIL void det_floatarr(void) {
        // Determinant of a square matrix
        *(float *)REF(ref_out) = aux_arr_det_float(REF(ref_arg0), REF(ref_arg2), *(int *)REF(ref_arg1));
        result_float_ref();
    }
    """


@norm_indent
def get_arr_interp_code() -> str:
    """The kernels are in aux_functions.c"""
    descriptions = {'lerp': 'Table lookup at a fractional index',
                    'bsearch': 'Fractional index of a value in a grid'}
    code = ''
    for name, description in descriptions.items():
        code += f"""
    STENCIL void {name}_floatarr_float(void) {{
        // {description}
        *(float *)REF(ref_out) = aux_arr_{name}_float(REF(ref_arg0), *(float *)REF(ref_arg1), *(int *)REF(ref_arg2));
        result_float_ref();
    }}

    STENCIL void {name}_floatarr_floatarr(void) {{
        // {description}, element-wise for an array of positions
        aux_arr_{name}_floatarr(REF(ref_arg0), REF(ref_arg1), REF(ref_out), *(int *)REF(ref_arg2), *(int *)REF(ref_arg3));
        result_float_ref();
    }}
    """
    return code


@norm_indent
def get_arr_func1_code(name: str, func: str, type1: str, type_out: str = 'float') -> str:
    kernel = f"aux_arr_{name}_{type1}"
    return f"""
    KERNEL void {kernel}(const {type1} *restrict a, {type_out} *restrict o, int n) {{
        for (int i = 0; i < n; i++) o[i] = {func}({arr_casted_operand(type1, 'a[i]', type_out)});
    }}

    STENCIL void {name}_{type1}arr(void) {{
        // Element-wise {name} of {type1} array
        {kernel}(REF(ref_arg0), REF(ref_out), *(int *)REF(ref_arg1));
        result_{type_out}_ref();
    }}
    """


@norm_indent
def get_arr_cast_code(name: str, type1: str, expr: str, type_out: str) -> str:
    """Element-wise type conversion of an array, expr converts the element a[i]"""
    kernel = f"aux_arr_{name}_{type1}"
    comment = f"// Element-wise conversion of {type1} array to {name}"
    if name == 'byte':
        comment += " with saturation, nan is converted to 0"
    return f"""
    KERNEL void {kernel}(const {type1} *restrict a, {type_out} *restrict o, int n) {{
        for (int i = 0; i < n; i++) o[i] = {expr};
    }}

    STENCIL void {name}_{type1}arr(void) {{
        {comment}
        {kernel}(REF(ref_arg0), REF(ref_out), *(int *)REF(ref_arg1));
        result_{type_out}_ref();
    }}
    """


@norm_indent
def get_arr_copy_code() -> str:
    """The kernels aux_arr_copy32 and aux_arr_copy8 are in aux_functions.c"""
    return """
    STENCIL void copy32_arr(void) {
        // Strided copy of 32 bit elements
        aux_arr_copy32(REF(ref_arg0), REF(ref_out), REF(ref_arg1));
        result_void();
    }

    STENCIL void copy8_arr(void) {
        // Strided copy of bytes
        aux_arr_copy8(REF(ref_arg0), REF(ref_out), REF(ref_arg1));
        result_void();
    }
    """


@norm_indent
def get_arr_ring_code() -> str:
    """Ring buffer of 32 bit elements, the kernel aux_arr_roll32 is in aux_functions.c"""
    return """
    SHORT_LOOP STENCIL void put32_arr(void) {
        // Writes the m elements of ref_arg1 in place to the elements i * m to i * m + m - 1
        // of a ring buffer (ref_out is the memory of the buffer ref_arg0, i = ref_arg2, m = ref_arg3)
        const int *a = REF(ref_arg1);
        int m = *(int *)REF(ref_arg3);
        int *o = (int *)REF(ref_out) + *(int *)REF(ref_arg2) * m;
        for (int i = 0; i < m; i++) o[i] = a[i];
        result_void();
    }

    STENCIL void roll32_arr(void) {
        // Copies the ring buffer ref_arg0 of n elements (ref_arg2) starting at the element ref_arg1
        aux_arr_roll32(REF(ref_arg0), REF(ref_out), *(int *)REF(ref_arg1), *(int *)REF(ref_arg2));
        result_void();
    }
    """


@norm_indent
def get_arr_conv_code() -> str:
    """The kernel aux_arr_conv2d is in aux_functions.c"""
    return """
    STENCIL void conv2d_floatarr_floatarr(void) {
        // 2D convolution
        aux_arr_conv2d(REF(ref_arg0), REF(ref_arg1), REF(ref_arg2), REF(ref_out), REF(ref_arg3));
        result_float_ref();
    }
    """


@norm_indent
def get_arr_pool_code(op: str) -> str:
    """2D pooling: largest element (max) or mean (avg) of each window"""
    description = ('largest element of each window, the padding is ignored' if op == 'max' else
                   'mean of each window, the padding counts as zeros')
    init = '-__builtin_inff()' if op == 'max' else '0.0f'
    accumulate = 'float v = xr[ox * sw - tx]; r[ox] = v > r[ox] ? v : r[ox];' if op == 'max' else 'r[ox] += xr[ox * sw - tx];'
    scale = '' if op == 'max' else 'for (int i = 0; i < oh * ow; i++) oc[i] *= 1.0f / (float)(kh * kw);'
    return f"""
    KERNEL void aux_arr_{op}pool2d(const float *restrict x, float *restrict o, const int *restrict p) {{
        // Like for the convolution each window element is accumulated
        // over the output rows and the loop bounds exclude the padding
        int n = p[0], h = p[1], w = p[2], kh = p[3], kw = p[4], oh = p[5], ow = p[6];
        int sh = p[7], sw = p[8], ph = p[9], pw = p[10];
        for (int c = 0; c < n; c++) {{
            const float *xc = x + c * h * w;
            float *restrict oc = o + c * oh * ow;
            for (int i = 0; i < oh * ow; i++) oc[i] = {init};
            for (int ky = 0; ky < kh; ky++) {{
                // Input row of output row oy: oy * sh - ty, must be in [0, h)
                int ty = ph - ky;
                int y0 = ty > 0 ? (ty + sh - 1) / sh : 0;
                int y1 = h + ty > 0 ? (h + ty - 1) / sh + 1 : 0;
                if (y1 > oh) y1 = oh;
                for (int kx = 0; kx < kw; kx++) {{
                    int tx = pw - kx;
                    int x0 = tx > 0 ? (tx + sw - 1) / sw : 0;
                    int x1 = w + tx > 0 ? (w + tx - 1) / sw + 1 : 0;
                    if (x1 > ow) x1 = ow;
                    for (int oy = y0; oy < y1; oy++) {{
                        const float *xr = xc + (oy * sh - ty) * w;
                        float *restrict r = oc + oy * ow;
                        for (int ox = x0; ox < x1; ox++) {{ {accumulate} }}
                    }}
                }}
            }}
            {scale}
        }}
    }}

    STENCIL void {op}pool2d_floatarr(void) {{
        // 2D {op} pooling of a float input [n, h, w] (batch and channels as n):
        // {description}. Parameters as int array [n, h, w, kh, kw, oh, ow, sh, sw, ph, pw]
        // with the window size k, the output size o, the strides s and the padding p
        aux_arr_{op}pool2d(REF(ref_arg0), REF(ref_out), REF(ref_arg1));
        result_float_ref();
    }}
    """


def get_sort_network(compare_exchange: str) -> str:
    """Loops of Batcher's odd-even merge sort network for n elements"""
    return f"""// Batcher's odd-even merge sort network: the sequence of compare-exchanges
        // (indices ia < ib) only depends on n, not on the data. With a branch-free
        // compare-exchange the execution time is constant for a given n.
        for (int p = 1; p < n; p += p) {{
            int mask = ~(2 * p - 1);
            for (int k = p; k >= 1; k /= 2) {{
                for (int j = k & (p - 1); j + k < n; j += 2 * k) {{  // k % p, p is a power of 2
                    int last = (k - 1 < n - j - k - 1) ? k - 1 : n - j - k - 1;
                    for (int i = 0; i <= last; i++) {{
                        int ia = i + j, ib = i + j + k;
                        if ((ia & mask) == (ib & mask)) {{
                            {compare_exchange}
                        }}
                    }}
                }}
            }}
        }}"""


@norm_indent
def get_arr_sort_code(type1: str) -> str:
    """Sorted copy (sort) and sorting indices (argsort) of an array by a sorting network"""
    if type1 == 'float':
        sort_cx = ('union { float f; int i; } x = {o[ia]}, y = {o[ib]}; '
                   'int m = -(x.f > y.f); VALUE_BARRIER(m); '
                   'int d = (x.i ^ y.i) & m; x.i ^= d; y.i ^= d; o[ia] = x.f; o[ib] = y.f;')
    else:
        sort_cx = ('int x = o[ia], y = o[ib]; int m = -(x > y); VALUE_BARRIER(m); '
                   'int d = (x ^ y) & m; o[ia] = x ^ d; o[ib] = y ^ d;')
    arg_cx = (f'int xa = o[ia], xb = o[ib]; {type1} va = a[xa], vb = a[xb]; '
              'int m = -((va > vb) | ((va == vb) & (xa > xb))); VALUE_BARRIER(m); '
              'int d = (xa ^ xb) & m; o[ia] = xa ^ d; o[ib] = xb ^ d;')
    return f"""
    KERNEL void aux_arr_sort_{type1}(const {type1} *restrict a, {type1} *restrict o, int n) {{
        // Swap by a mask from the comparison result: no data dependent branches, also on
        // targets without min/max instructions (a?b:c is compiled to a branch by gcc). The
        // barrier on the mask keeps gcc from converting the masked swap into a branch.
        for (int i = 0; i < n; i++) o[i] = a[i];
        {get_sort_network(sort_cx)}
    }}

    KERNEL void aux_arr_argsort_{type1}(const {type1} *restrict a, int *restrict o, int n) {{
        // Order by (value, index): deterministic result equal to a stable sort,
        // swap by a mask like for the sort
        for (int i = 0; i < n; i++) o[i] = i;
        {get_sort_network(arg_cx)}
    }}

    STENCIL void sort_{type1}arr(void) {{
        // Sorted copy of {type1} array
        aux_arr_sort_{type1}(REF(ref_arg0), REF(ref_out), *(int *)REF(ref_arg1));
        result_{type1}_ref();
    }}

    STENCIL void argsort_{type1}arr(void) {{
        // Indices that sort a {type1} array
        aux_arr_argsort_{type1}(REF(ref_arg0), REF(ref_out), *(int *)REF(ref_arg1));
        result_int_ref();
    }}
    """


@norm_indent
def get_arr_sum_code(type1: str) -> str:
    kernel = f"aux_arr_sum_{type1}"
    body = get_reduction_body(type1, lambda i: f"a[{i}]")
    return f"""
    KERNEL {type1} {kernel}(const {type1} *restrict a, int n) {{
        {body}
    }}

    STENCIL void sum_{type1}arr(void) {{
        // Sum of all elements of {type1} array
        *({type1} *)REF(ref_out) = {kernel}(REF(ref_arg0), *(int *)REF(ref_arg1));
        result_{type1}_ref();
    }}
    """


@norm_indent
def get_arr_minmax_code(op: str, type1: str, lanes: int = 8) -> str:
    kernel = f"aux_arr_{op}_{type1}"
    cmp = '<' if op == 'min' else '>'
    description = 'Smallest' if op == 'min' else 'Largest'
    decl = ', '.join(f'r{k} = a[0]' for k in range(lanes))
    body = ' '.join(f'r{k} = a[i + {k}] {cmp} r{k} ? a[i + {k}] : r{k};' for k in range(lanes))
    total = ' '.join(f'r = r{k} {cmp} r ? r{k} : r;' for k in range(lanes))
    return f"""
    KERNEL {type1} {kernel}(const {type1} *restrict a, int n) {{
        // Independent partial results allow vectorization like for the sum
        {type1} {decl}, r = a[0];
        int i = 0;
        for (; i + {lanes} <= n; i += {lanes}) {{
            {body}
        }}
        for (; i < n; i++) r = a[i] {cmp} r ? a[i] : r;
        {total}
        return r;
    }}

    STENCIL void {op}_{type1}arr(void) {{
        // {description} element of {type1} array
        *({type1} *)REF(ref_out) = {kernel}(REF(ref_arg0), *(int *)REF(ref_arg1));
        result_{type1}_ref();
    }}
    """


@norm_indent
def get_arr_sumaxis_code(type1: str) -> str:
    return f"""
    KERNEL void aux_arr_sumaxis_{type1}(const {type1} *restrict a, {type1} *restrict o, const int *restrict p) {{
        // Like for the matrix product the inner loop is contiguous,
        // the first k step initializes the output
        int m = p[0], k = p[1], n = p[2];
        for (int i = 0; i < m; i++) {{
            {type1} *restrict r = o + i * n;
            const {type1} *ar = a + i * k * n;
            for (int j = 0; j < n; j++) r[j] = ar[j];
            for (int q = 1; q < k; q++) {{
                const {type1} *aq = ar + q * n;
                for (int j = 0; j < n; j++) r[j] += aq[j];
            }}
        }}
    }}

    STENCIL void sumaxis_{type1}arr(void) {{
        // Sum along the middle axis of a row-major array [m, k, n], the dimensions
        // are passed as int array. The result is an array [m, n].
        aux_arr_sumaxis_{type1}(REF(ref_arg0), REF(ref_out), REF(ref_arg1));
        result_{type1}_ref();
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
    ops = ['add', 'sub', 'mul', 'div', 'floordiv', 'gt', 'ge', 'eq', 'ne']
    int_ops = ['bwand', 'bwor', 'bwxor', 'lshift', 'rshift']

    for t1 in types:
        code += get_result_stubs1(t1)

    for t1, t2 in permutate(types, types):
        code += get_result_stubs2(t1, t2)

    code += get_entry_function_shell()

    code += get_cast('float', 'int')
    code += get_cast('int', 'float')

    for fn in ['get_42', 'tanh']:
        code += get_func1(fn, 'float')

    for t in types:
        code += get_neg(t)

    for t in types:
        code += get_custom_stencil(f"square_{t}({t} arg1)", f"result_{t}(arg1 * arg1);")

    fnames = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan']
    for fn in fnames:
        code += get_math_func1(fn + 'f', 'float', fn)

    code += get_math_func1('fabsf', 'float', 'abs')
    code += get_custom_stencil('abs_int(int arg1)', 'result_int(__builtin_abs(arg1));')

    for t in types:
        code += get_custom_stencil(f"sign_{t}({t} arg1)", "result_int((arg1 > 0) - (arg1 < 0));")

    for fn in ['atan2', 'pow']:
        code += get_math_func2(fn, 'float', 'float')
    code += get_custom_stencil('pow_int_int(int arg1, int arg2)', 'result_int_int(aux_ipow(arg1, arg2), arg2);')

    # Branch-free selection
    gtabs_comment = 'Branch-free selection: int mask -1 if |arg1| > |arg2| else 0'
    mask_comment = 'Branch-free selection: arg1 where the mask arg2 is set, otherwise 0'
    masknot_comment = 'Branch-free selection: arg1 where the mask arg2 is not set, otherwise 0'
    code += get_custom_stencil('gtabs_float_float(float arg1, float arg2)',
                               'result_int_float(-(fabsf(arg1) > fabsf(arg2)), arg2);', gtabs_comment)
    code += get_custom_stencil('mask_float_int(float arg1, int arg2)',
                               'union { float f; int i; } x = {arg1}; x.i &= arg2; result_float_int(x.f, arg2);', mask_comment)
    code += get_custom_stencil('masknot_float_int(float arg1, int arg2)',
                               'union { float f; int i; } x = {arg1}; x.i &= ~arg2; result_float_int(x.f, arg2);', masknot_comment)
    code += get_custom_stencil('mask_int_int(int arg1, int arg2)', 'result_int_int(arg1 & arg2, arg2);', mask_comment)
    code += get_custom_stencil('masknot_int_int(int arg1, int arg2)', 'result_int_int(arg1 & ~arg2, arg2);', masknot_comment)

    for t in types:
        code += get_min(t, t)
        code += get_max(t, t)

    for op, t in permutate(ops, types):
        if op == 'floordiv':
            code += get_floordiv('floordiv', t, t)
        elif op == 'div':
            if t == 'float':
                code += get_op_code_float(op, t, t)
        elif op in {'gt', 'eq', 'ge', 'ne'}:
            code += get_op_code(op, t, t, 'int')
        else:
            code += get_op_code(op, t, t, t)

    for op in int_ops:
        code += get_op_code(op, 'int', 'int', 'int')

    code += get_op_code('mod', 'int', 'int', 'int')

    for t1, t2, t_out in permutate(types, types, types):
        code += get_load_reg0_code(t1, t2, t_out)
        code += get_load_reg1_code(t1, t2, t_out)

    for t1, t2 in permutate(types, types):
        code += get_store_code(t1, t2)

    # Array stencils:
    # The result stub gives the element type of the result written to ref_out,
    # result_void for results of the type of the arguments (strided copy)
    code += get_result_stubs1('void').replace('void arg1', 'void')
    for t in types:
        code += f"void result_{t}_ref(void);\n"
    code += "void result_byte_ref(void);\n"

    for op, t1, t2 in permutate(['add', 'sub', 'mul', 'div'], types, types):
        code += get_arr_op_code(op, t1, t2)

    for op, t1, t2 in permutate(['min', 'max', 'gt', 'ge', 'eq', 'ne'], types, types):
        code += get_arr_op_code(op, t1, t2)

    for fn, t1, t2 in permutate(['pow', 'atan2'], types, types):
        int_pow = fn == 'pow' and t1 == t2 == 'int'
        code += get_arr_op_code(fn, t1, t2, 'aux_ipow' if int_pow else fn + 'f')

    fnames = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan']
    for fn, t in permutate(fnames, types):
        code += get_arr_func1_code(fn, fn + 'f', t)

    for t in types:
        code += get_arr_func1_code('tanh', 'aux_tanh', t)

    code += get_arr_func1_code('abs', 'fabsf', 'float')
    code += get_arr_func1_code('abs', '__builtin_abs', 'int', 'int')

    for t1, t2 in permutate(types, types):
        code += get_arr_dot_code(t1, t2)
        code += get_arr_matmul_code(t1, t2)

    code += get_arr_solve_code()
    code += get_arr_interp_code()

    code += get_arr_cast_code('float', 'int', '(float)a[i]', 'float')
    code += get_arr_cast_code('int', 'float', '(int)a[i]', 'int')
    for t in types:
        code += get_arr_cast_code('bool', t, 'a[i] != 0', 'int')

    # Byte arrays: converted to int or float for operations
    code += get_arr_cast_code('float', 'byte', '(float)a[i]', 'float')
    code += get_arr_cast_code('int', 'byte', '(int)a[i]', 'int')
    code += get_arr_cast_code('byte', 'int', '(byte)(a[i] > 255 ? 255 : a[i] > 0 ? a[i] : 0)', 'byte')
    code += get_arr_cast_code('byte', 'float', '(byte)(a[i] > 255.0f ? 255.0f : a[i] > 0.0f ? a[i] : 0.0f)', 'byte')

    code += get_arr_copy_code()
    code += get_arr_ring_code()
    code += get_arr_conv_code()
    code += get_arr_pool_code('max')
    code += get_arr_pool_code('avg')

    for t in types:
        code += get_arr_sum_code(t)
        code += get_arr_sumaxis_code(t)
        code += get_arr_minmax_code('min', t)
        code += get_arr_minmax_code('max', t)
        code += get_arr_sort_code(t)

    print(f"Write file {args.path}...")
    with open(args.path, 'w') as f:
        f.write(code)
