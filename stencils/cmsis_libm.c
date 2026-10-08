// Single precision math functions for Cortex-M targets based on CMSIS-DSP.
// Replaces the corresponding MUSL functions, which use double arithmetic
// (software emulated on single precision FPUs like FPv4-SP and FPv5-SP).
//
// sin, cos and atan2 are CMSIS-DSP functions (Apache-2.0), exp and log are
// scalar versions of vexpq_f32 and vlogq_f32 from the ComputeLibrary part of
// CMSIS-DSP (NEMath.h, Copyright (c) 2016, 2019 ARM Limited, MIT license).
// tan uses the polynomial of tanf from the Cephes Math Library (Copyright
// 1984, 1987, 1989, 1992 by Stephen L. Moshier).
//
// Math functions of the Cortex-M stencils (errors measured by test.c):
//
//   Function        Implementation                              Max. error
//   --------------  ------------------------------------------  ------------------
//   sinf, cosf      arm_sin_f32, arm_cos_f32 (table with        2e-5 absolute
//                   linear interpolation)
//   atanf, atan2f   arm_atan2_f32 (polynomial)                  3e-7 absolute
//   asinf, acosf    arm_atan2_f32 and sqrtf                     3e-7 absolute
//   expf            polynomial of vexpq_f32                     4e-6 relative
//   logf            polynomial of vlogq_f32                     6e-6 absolute
//   powf            expf(y * logf(x))                           3e-5 relative
//                                                               (for |y| <= 6)
//   tanf            polynomial of the Cephes tanf               3e-7 relative
//                                                               (for |x| <= 2000)
//   sqrtf, floorf   MUSL, single precision only                 exact

#include <stdint.h>
#include <math.h>
#include "dsp/fast_math_functions.h"

// Prefix for the function names, for testing against the C library on a host
#ifndef CM_NAME
#define CM_NAME(f) f
#endif

#define CM_PI 3.14159265358979f
#define CM_LN2 0.6931471805f
#define CM_INV_LN2 1.4426950408f

// sin and cos are only valid for arguments which can be converted to int32_t
// after scaling with 1/(2*pi), this excludes also inf and NaN. Larger arguments
// have no significant digits left.
#define CM_TRIG_MAX 1.0e9f

typedef union { float f; int32_t i; uint32_t u; } cm_float_bits;

// Coefficients p0, p4, p2, p6, p1, p5, p3, p7 like exp_tab and log_tab
static const float cm_exp_tab[8] = {
    1.f, 0.0416598916054f, 0.500000596046f, 0.0014122662833f,
    1.00000011921f, 0.00833693705499f, 0.166665703058f, 0.000195780929062f
};

static const float cm_log_tab[8] = {
    -2.29561495781f, -2.47071170807f, -5.68692588806f, -0.165253549814f,
    5.17591238022f, 0.844007015228f, 4.58445882797f, 0.0141278216615f
};

static inline float cm_taylor_poly(float x, const float *coeffs) {
    float a = coeffs[0] + coeffs[4] * x;
    float b = coeffs[2] + coeffs[6] * x;
    float c = coeffs[1] + coeffs[5] * x;
    float d = coeffs[3] + coeffs[7] * x;
    float x2 = x * x;
    float x4 = x2 * x2;
    return (a + b * x2) + (c + d * x2) * x4;
}

// Reduction to [-pi, pi] with 2*pi split into two parts. The product of n and
// the first part is exact for |x| < 4e5, arm_sin_f32 and arm_cos_f32 reduce
// the argument by scaling it with 1/(2*pi) which loses precision for large x.
static inline float cm_reduce_2pi(float x) {
    float n = (float)(int32_t)(x * 0.159154943092f + (x < 0.0f ? -0.5f : 0.5f));
    return (x - n * 6.28125f) - n * 1.9353071795864769e-3f;
}

float CM_NAME(sinf)(float x) {
    if (!(fabsf(x) < CM_TRIG_MAX)) return x - x;
    return arm_sin_f32(cm_reduce_2pi(x));
}

float CM_NAME(cosf)(float x) {
    if (!(fabsf(x) < CM_TRIG_MAX)) return x - x;
    return arm_cos_f32(cm_reduce_2pi(x));
}

// Reduction to [-pi/4, pi/4] with pi/2 split into three parts, polynomial of
// the Cephes tanf and -1/tan(r) for the odd multiples of pi/2. The reduction
// is accurate for |x| < 2000, for larger arguments the error increases close
// to the poles.
float CM_NAME(tanf)(float x) {
    if (!(fabsf(x) < CM_TRIG_MAX)) return x - x;
    int32_t n = (int32_t)(x * 0.636619772367581f + (x < 0.0f ? -0.5f : 0.5f));
    float fn = (float)n;
    float r = ((x - fn * 1.5703125f) - fn * 4.837512969970703125e-4f) - fn * 7.54978995489188216e-8f;
    float z = r * r;
    float y = r + r * z * (3.33331568548e-1f + z * (1.33387994085e-1f + z * (5.34112807005e-2f +
              z * (2.44301354525e-2f + z * (3.11992232697e-3f + z * 9.38540185543e-3f)))));
    return n & 1 ? -1.0f / y : y;
}

float CM_NAME(atan2f)(float y, float x) {
    float result;
    if (arm_atan2_f32(y, x, &result) == ARM_MATH_SUCCESS) return result;
    // Not handled by arm_atan2_f32: NaN and y = x = 0
    if (x != x || y != y) return x + y;
    return signbit(x) ? copysignf(CM_PI, y) : y;
}

float CM_NAME(atanf)(float x) {
    return CM_NAME(atan2f)(x, 1.0f);
}

float CM_NAME(asinf)(float x) {
    return CM_NAME(atan2f)(x, sqrtf((1.0f - x) * (1.0f + x)));
}

float CM_NAME(acosf)(float x) {
    return CM_NAME(atan2f)(sqrtf((1.0f - x) * (1.0f + x)), x);
}

float CM_NAME(expf)(float x) {
    if (!(x < 88.72f)) return x != x ? x : __builtin_inff();
    if (x < -87.33f) return 0.0f;

    // Range reduction to [-ln(2), ln(2)]
    int32_t m = (int32_t)(x * CM_INV_LN2);
    cm_float_bits r;
    r.f = cm_taylor_poly(x - (float)m * CM_LN2, cm_exp_tab);

    // Multiply with 2^m
    r.i += m * (1 << 23);
    return r.f;
}

float CM_NAME(logf)(float x) {
    cm_float_bits v = {x};
    if (!(x > 0.0f)) return x == 0.0f ? -__builtin_inff() : __builtin_nanf("");
    if (v.u >= 0x7f800000u) return x;  // inf
    int32_t e = 0;
    if (v.u < 0x00800000u) {  // subnormal
        v.f = x * 8388608.0f;
        e = -23;
    }

    // Split into exponent and mantissa in [1, 2)
    int32_t m = (int32_t)(v.u >> 23) - 127;
    v.i -= m * (1 << 23);

    return cm_taylor_poly(v.f, cm_log_tab) + (float)(m + e) * CM_LN2;
}

float CM_NAME(powf)(float x, float y) {
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (x != x || y != y) return x + y;
    if (x > 0.0f) return CM_NAME(expf)(y * CM_NAME(logf)(x));

    // x <= 0: only defined for integer exponents, odd ones keep the sign
    float yi = floorf(y);
    if (x == 0.0f) return y > 0.0f ? 0.0f : __builtin_inff();
    if (yi != y) return __builtin_nanf("");
    float r = CM_NAME(expf)(y * CM_NAME(logf)(-x));
    return fabsf(y) < 16777216.0f && ((int32_t)yi & 1) ? -r : r;
}
