#include <stdio.h>
#include "aux_functions.c"

// The math functions of the Cortex-M stencils are tested with TEST_CMSIS_LIBM,
// which requires CMSIS-DSP (see tools/test_stencil_aux.sh)
#ifdef TEST_CMSIS_LIBM
#define CM_NAME(f) cm_##f
#include "cmsis_libm.c"
#endif

static int errors = 0;

#ifdef TEST_CMSIS_LIBM
// Maximum of the error relative to the tolerance max(rel * |ref|, abs_tol)
static void check(const char *name, double res, double ref, double rel, double abs_tol, double *worst) {
    if (ref != ref) {
        if (res == res) {
            printf("%s: %g, expected nan\n", name, res);
            errors++;
        }
        return;
    }
    double tol = fmax(rel * fabs(ref), abs_tol);
    double e = res == ref ? 0.0 : fabs(res - ref) / tol;
    if (!(e <= 1.0)) {
        if (errors < 20) printf("%s: %.9g, expected %.9g\n", name, res, ref);
        errors++;
    }
    if (e > *worst) *worst = e;
}

// CMSIS-DSP based math functions against the C library
static void test_cmsis_libm(void) {
    double w_sin = 0, w_cos = 0, w_atan = 0, w_atan2 = 0, w_asin = 0, w_acos = 0;
    double w_exp = 0, w_log = 0, w_pow = 0, w_special = 0, w_sin_large = 0, w_tan = 0;

    // Table based sin and cos: absolute error up to 2e-5
    for (int i = -200000; i <= 200000; i++) {
        float x = (float)i * 0.0001f;
        check("sinf", cm_sinf(x), sin((double)x), 0, 2.5e-5, &w_sin);
        check("cosf", cm_cosf(x), cos((double)x), 0, 2.5e-5, &w_cos);
        check("atanf", cm_atanf(x), atan((double)x), 1e-5, 1e-6, &w_atan);
        check("atan2f", cm_atan2f(x, -0.7f), atan2((double)x, -0.7), 1e-5, 1e-6, &w_atan2);
        check("atan2f", cm_atan2f(0.3f, x), atan2(0.3, (double)x), 1e-5, 1e-6, &w_atan2);
        check("expf", cm_expf(x * 4.0f), exp((double)(x * 4.0f)), 1e-5, 1e-37, &w_exp);
    }

    // tan with reduction in single precision, accurate for |x| < 2000
    for (int i = -200000; i <= 200000; i++) {
        float x = (float)i * 0.01f;
        check("tanf", cm_tanf(x), tan((double)x), 1e-6, 1e-30, &w_tan);
    }

    // Large arguments
    for (int i = -200000; i <= 200000; i++) {
        float x = (float)i * 1.7f;
        check("sinf", cm_sinf(x), sin((double)x), 0, 5e-5, &w_sin_large);
        check("cosf", cm_cosf(x), cos((double)x), 0, 5e-5, &w_sin_large);
    }

    for (int i = -10000; i <= 10000; i++) {
        float x = (float)i * 0.0001f;
        check("asinf", cm_asinf(x), asin((double)x), 1e-5, 1e-6, &w_asin);
        check("acosf", cm_acosf(x), acos((double)x), 1e-5, 1e-6, &w_acos);
    }

    for (float x = 1e-37f; x < 1e38f; x *= 1.0001f)
        check("logf", cm_logf(x), log((double)x), 1e-5, 1e-5, &w_log);

    for (float x = 0.01f; x < 100.0f; x *= 1.07f)
        for (float y = -6.0f; y <= 6.0f; y += 0.37f)
            check("powf", cm_powf(x, y), pow((double)x, (double)y), 5e-5, 1e-30, &w_pow);

    // Special values
    float inf = __builtin_inff(), nan = __builtin_nanf("");
    check("sinf(inf)", cm_sinf(inf), nan, 0, 0, &w_special);
    check("cosf(nan)", cm_cosf(nan), nan, 0, 0, &w_special);
    check("sinf(1e30)", cm_sinf(1e30f), 0, 0, 1, &w_special);
    check("tanf(inf)", cm_tanf(inf), nan, 0, 0, &w_special);
    check("tanf(nan)", cm_tanf(nan), nan, 0, 0, &w_special);
    check("tanf(100000)", cm_tanf(100000.0f), tan(100000.0), 1e-5, 1e-5, &w_special);
    check("atan2f(0, 0)", cm_atan2f(0.0f, 0.0f), 0, 0, 0, &w_special);
    check("atan2f(0, -0)", cm_atan2f(0.0f, -0.0f), M_PI, 1e-6, 0, &w_special);
    check("atan2f(-0, -0)", cm_atan2f(-0.0f, -0.0f), -M_PI, 1e-6, 0, &w_special);
    check("atan2f(nan, 1)", cm_atan2f(nan, 1.0f), nan, 0, 0, &w_special);
    check("atan2f(1, inf)", cm_atan2f(1.0f, inf), 0, 0, 0, &w_special);
    check("atan2f(inf, 1)", cm_atan2f(inf, 1.0f), M_PI / 2, 1e-6, 0, &w_special);
    check("asinf(1)", cm_asinf(1.0f), M_PI / 2, 1e-6, 0, &w_special);
    check("asinf(2)", cm_asinf(2.0f), nan, 0, 0, &w_special);
    check("acosf(-1)", cm_acosf(-1.0f), M_PI, 1e-6, 0, &w_special);
    check("expf(100)", cm_expf(100.0f), inf, 0, 0, &w_special);
    check("expf(-100)", cm_expf(-100.0f), 0, 0, 0, &w_special);
    check("expf(-inf)", cm_expf(-inf), 0, 0, 0, &w_special);
    check("expf(nan)", cm_expf(nan), nan, 0, 0, &w_special);
    check("expf(0)", cm_expf(0.0f), 1, 1e-6, 0, &w_special);
    check("logf(0)", cm_logf(0.0f), -inf, 0, 0, &w_special);
    check("logf(-1)", cm_logf(-1.0f), nan, 0, 0, &w_special);
    check("logf(inf)", cm_logf(inf), inf, 0, 0, &w_special);
    check("logf(1e-42)", cm_logf(1e-42f), log((double)1e-42f), 1e-5, 0, &w_special);
    check("powf(10, 0)", cm_powf(10.0f, 0.0f), 1, 0, 0, &w_special);
    check("powf(0, 2)", cm_powf(0.0f, 2.0f), 0, 0, 0, &w_special);
    check("powf(0, -1)", cm_powf(0.0f, -1.0f), inf, 0, 0, &w_special);
    check("powf(-2, 3)", cm_powf(-2.0f, 3.0f), -8, 1e-5, 0, &w_special);
    check("powf(-2, 2)", cm_powf(-2.0f, 2.0f), 4, 1e-5, 0, &w_special);
    check("powf(-2, 0.5)", cm_powf(-2.0f, 0.5f), nan, 0, 0, &w_special);

    printf("error / tolerance: sin %.2f, cos %.2f, sin and cos of large arguments %.2f, tan %.2f, atan %.2f, atan2 %.2f, asin %.2f, acos %.2f, exp %.2f, log %.2f, pow %.2f\n",
           w_sin, w_cos, w_sin_large, w_tan, w_atan, w_atan2, w_asin, w_acos, w_exp, w_log, w_pow);
}
#endif

int main() {
    // Test aux functions
    float g42 = aux_get_42(0.0f);
    if (g42 != 231.0f) {
        printf("aux_get_42(0) = %f, expected 231\n", g42);
        errors++;
    }

    // aux_tanh against tanhf of the C library, including the range of the Taylor series
    for (int i = -4000; i <= 4000; i++) {
        float x = (float)i * 0.0025f;
        float res = aux_tanh(x);
        float ref = tanhf(x);
        if (fabsf(res - ref) > 1e-6f) {
            printf("aux_tanh(%f) = %.9f, expected %.9f\n", x, res, ref);
            errors++;
        }
    }

    // aux_arr_solve_float: a zero on the diagonal requires pivoting, two right-hand sides
    const float sa[9] = {0.0f, 2.0f, 1.0f, 1.0f, 1.0f, 0.5f, 4.0f, -1.0f, 3.0f};
    const float sb[6] = {1.0f, 3.0f, 2.0f, 2.5f, 3.0f, 6.0f};
    const float sref[6] = {1.5f, 1.0f, 6.0f / 7.0f, 1.0f, -5.0f / 7.0f, 1.0f};
    const int sdims[2] = {3, 2};
    float sx[6], sw[9];
    aux_arr_solve_float(sa, sb, sx, sw, sdims);
    for (int i = 0; i < 6; i++) {
        if (fabsf(sx[i] - sref[i]) > 1e-5f) {
            printf("aux_arr_solve_float: x[%d] = %.9f, expected %.9f\n", i, sx[i], sref[i]);
            errors++;
        }
    }

    // aux_arr_det_float: one row swap, no swap, singular with a zero column and 1 x 1
    const float da[4][9] = {
        {0.0f, 2.0f, 1.0f, 1.0f, 1.0f, 0.5f, 4.0f, -1.0f, 3.0f},
        {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f},
        {1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 1.0f, -2.0f, 0.0f, 5.0f},
        {-2.5f}};
    const int dn[4] = {3, 3, 3, 1};
    const float dref[4] = {-7.0f, 18.0f, 0.0f, -2.5f};
    for (int t = 0; t < 4; t++) {
        float res = aux_arr_det_float(da[t], sw, dn[t]);
        if (!(fabsf(res - dref[t]) <= 1e-5f)) {
            printf("aux_arr_det_float: test %d = %.9f, expected %.9f\n", t, res, dref[t]);
            errors++;
        }
    }

    // aux_arr_bsearch_float and aux_arr_lerp_float: fractional index in a non-uniform
    // grid and lookup at this index, clamped outside of the grid, for all table sizes
    // up to 9 (number of search steps)
    const float gx[9] = {-2.0f, -1.5f, 0.0f, 0.25f, 1.0f, 3.0f, 3.5f, 7.0f, 8.0f};
    const float gf[9] = {1.0f, -1.0f, 4.0f, 2.0f, 2.0f, 0.5f, 8.0f, -3.0f, 6.0f};
    for (int n = 2; n <= 9; n++) {
        for (int k = -30; k <= 90; k++) {
            float x = (float)k * 0.125f;
            float ref_t = 0.0f, ref_f = gf[0];
            if (x >= gx[n - 1]) {
                ref_t = (float)(n - 1);
                ref_f = gf[n - 1];
            } else {
                for (int i = 0; i < n - 1; i++) {
                    if (x >= gx[i] && x < gx[i + 1]) {
                        float f = (x - gx[i]) / (gx[i + 1] - gx[i]);
                        ref_t = (float)i + f;
                        ref_f = gf[i] + f * (gf[i + 1] - gf[i]);
                    }
                }
            }
            float t = aux_arr_bsearch_float(gx, x, n);
            float res = aux_arr_lerp_float(gf, t, n);
            if (!(fabsf(t - ref_t) <= 1e-5f) || !(fabsf(res - ref_f) <= 1e-5f)) {
                if (errors < 20) printf("interp n=%d x=%f: index %.7f (%.7f), value %.7f (%.7f)\n", n, x, t, ref_t, res, ref_f);
                errors++;
            }
        }
    }

    // Exact table values at the grid points, nan and inf as position
    const float ti[6] = {-1e30f, 0.0f, 3.0f, 8.0f, 1e30f, INFINITY};
    const float tref[6] = {1.0f, 1.0f, 2.0f, 6.0f, 6.0f, 6.0f};
    for (int k = 0; k < 6; k++) {
        float res = aux_arr_lerp_float(gf, ti[k], 9);
        if (res != tref[k]) {
            printf("aux_arr_lerp_float(%g) = %.9f, expected %.9f\n", ti[k], res, tref[k]);
            errors++;
        }
    }
    float nan_res = aux_arr_lerp_float(gf, aux_arr_bsearch_float(gx, NAN, 9), 9);
    if (nan_res == nan_res || aux_arr_lerp_float(gf, -INFINITY, 9) != 1.0f ||
        aux_arr_bsearch_float(gx, INFINITY, 9) != 8.0f || aux_arr_bsearch_float(gx, -INFINITY, 9) != 0.0f) {
        printf("interp: wrong result for nan or inf\n");
        errors++;
    }

    // Array variants
    const float ax[4] = {-3.0f, 0.125f, 3.25f, 9.0f};
    float at[4], ar[4];
    aux_arr_bsearch_floatarr(gx, ax, at, 9, 4);
    aux_arr_lerp_floatarr(gf, at, ar, 9, 4);
    for (int k = 0; k < 4; k++) {
        if (at[k] != aux_arr_bsearch_float(gx, ax[k], 9) || ar[k] != aux_arr_lerp_float(gf, at[k], 9)) {
            printf("interp: array kernel differs from scalar kernel at %d\n", k);
            errors++;
        }
    }

#ifdef TEST_CMSIS_LIBM
    test_cmsis_libm();
#endif

    if (errors) {
        printf("%d errors\n", errors);
        return 1;
    }
    printf("aux functions ok\n");
    return 0;
}
