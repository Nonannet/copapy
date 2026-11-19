// Stencils with a single fixed signature, not generated for variants of
// types or functions. Included by generate_stencils.py, the include is removed.
#include "aux_functions.c"

// Result stubs used here (generate_stencils.py declares them for all types
// after the included files)
void result_int(int arg1);
void result_int_int(int arg1, int arg2);
void result_int_float(int arg1, float arg2);
void result_float_int(float arg1, int arg2);
void result_float_ref(void);
void result_void(void);

// Scalar stencils

STENCIL void abs_int(int arg1) {
    result_int(__builtin_abs(arg1));
}

STENCIL void pow_int_int(int arg1, int arg2) {
    result_int_int(aux_ipow(arg1, arg2), arg2);
}

// Branch-free selection

STENCIL void gtabs_float_float(float arg1, float arg2) {
    // Int mask -1 if |arg1| > |arg2| else 0
    result_int_float(-(fabsf(arg1) > fabsf(arg2)), arg2);
}

STENCIL void mask_float_int(float arg1, int arg2) {
    // arg1 where the mask arg2 is set, otherwise 0
    union { float f; int i; } x = {arg1};
    x.i &= arg2;
    result_float_int(x.f, arg2);
}

STENCIL void masknot_float_int(float arg1, int arg2) {
    // arg1 where the mask arg2 is not set, otherwise 0
    union { float f; int i; } x = {arg1};
    x.i &= ~arg2;
    result_float_int(x.f, arg2);
}

STENCIL void mask_int_int(int arg1, int arg2) {
    // arg1 where the mask arg2 is set, otherwise 0
    result_int_int(arg1 & arg2, arg2);
}

STENCIL void masknot_int_int(int arg1, int arg2) {
    // arg1 where the mask arg2 is not set, otherwise 0
    result_int_int(arg1 & ~arg2, arg2);
}

// Int division and modulo rounded towards -inf (Python semantics). The
// division instruction traps on some architectures (e.g. x86) for b = 0 and
// for INT_MIN / -1: it is never executed with these operands. Both are
// divided by 1 instead: x // 0 is 0 (x % 0 is x) and x // -1 is -x, wrapping
// around for INT_MIN. The variants for divisors known at compile time (not
// 0, -1, 1 or a power of 2, these are no divisions) need no handling of the
// trapping operands. All branch-free.

STENCIL void floordiv_int_int(int a, int b) {
    int m1 = b == -1;
    int d = b + (b == 0) + 2 * m1;
    int q = a / d;
    int r = a % d;
    q = (int)(((unsigned)q ^ -(unsigned)m1) + (unsigned)m1);
    q -= (r != 0) & ((r < 0) != (d < 0));
    result_int_int(q & -(b != 0), b);
}

STENCIL void mod_int_int(int a, int b) {
    int d = b + (b == 0) + 2 * (b == -1);
    int r = a % d;
    r += d & -((r != 0) & ((r < 0) != (d < 0)));
    int nz = -(b != 0);
    result_int_int((r & nz) | (a & ~nz), b);
}

STENCIL void floordivc_int_int(int a, int b) {
    // Divisor is a constant other than 0 and -1
    int r = a % b;
    result_int_int(a / b - ((r != 0) & ((r < 0) != (b < 0))), b);
}

STENCIL void modc_int_int(int a, int b) {
    // Divisor is a constant other than 0 and -1
    int r = a % b;
    result_int_int(r + (b & -((r != 0) & ((r < 0) != (b < 0)))), b);
}

// Array stencils, the kernels are in aux_functions.c

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

STENCIL void svd_floatarr(void) {
    // Singular value decomposition, the result is u, s and vh packed
    aux_arr_svd_float(REF(ref_arg0), REF(ref_out), REF(ref_arg2), REF(ref_arg1));
    result_float_ref();
}

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

STENCIL void conv2d_floatarr_floatarr(void) {
    // 2D convolution
    aux_arr_conv2d(REF(ref_arg0), REF(ref_arg1), REF(ref_arg2), REF(ref_out), REF(ref_arg3));
    result_float_ref();
}
