#include "stencil_helper.h"

const float PI      = 3.14159265358979323846f;
const float PI_2    = 1.57079632679489661923f;  // pi/2
const float TWO_OVER_PI = 0.63661977236758134308f; // 2/pi

NOINLINE float aux_sin(float x) {
    // convert to double for reduction (better precision)
    double xd = (double)x;

    // quadrant index q = nearest integer to x * 2/pi
    double qd = xd * (double)TWO_OVER_PI;
    // round to nearest integer (tie to even rounding not guaranteed)
    int q = (int)(qd + (qd >= 0.0 ? 0.5 : -0.5));

    // range-reduced remainder r = x − q*(pi/2)
    // use hi/lo parts for pi/2 to reduce error
    const double PIO2_HI = 1.57079625129699707031;     // ≈ first 24 bits
    const double PIO2_LO = 7.54978941586159635335e-08; // remainder
    double r_d = xd - (double)q * PIO2_HI - (double)q * PIO2_LO;
    float r = (float)r_d;

    // Select function and sign based on quadrant
    int qm = q & 3;
    int use_cos = (qm == 1 || qm == 3);
    int sign = (qm == 0 || qm == 1) ? +1 : -1;

    float r2 = r * r;

    if (!use_cos) {
        // sin(r) polynomial: r + s3*r^3 + s5*r^5 + s7*r^7 + s9*r^9
        const float s3 = -1.6666667163e-1f;
        const float s5 =  8.3333337680e-3f;
        const float s7 = -1.9841270114e-4f;
        const float s9 =  2.7557314297e-6f;

        float p = ((s9 * r2 + s7) * r2 + s5) * r2 + s3;
        float result = r + r * r2 * p;
        return sign * result;
    } else {
        // cos(r) polynomial: 1 + c2*r2 + c4*r4 + c6*r6 + c8*r8
        const float c2 = -0.5f;
        const float c4 =  4.1666667908e-2f;
        const float c6 = -1.3888889225e-3f;
        const float c8 =  2.4801587642e-5f;

        float p = ((c8 * r2 + c6) * r2 + c4) * r2 + c2;
        float result = 1.0f + r2 * p;
        return sign * result;
    }
}

NOINLINE float aux_cos(float x) {
    // convert to double for reduction (better precision)
    double xd = (double)x;

    // quadrant index q = nearest integer to x * 2/pi
    double qd = xd * (double)TWO_OVER_PI;
    // round to nearest integer (tie to even rounding not guaranteed)
    int q = (int)(qd + (qd >= 0.0 ? 0.5 : -0.5));

    // range-reduced remainder r = x − q*(pi/2)
    // use hi/lo parts for pi/2 to reduce error
    const double PIO2_HI = 1.57079625129699707031;     // ≈ first 24 bits
    const double PIO2_LO = 7.54978941586159635335e-08; // remainder
    double r_d = xd - (double)q * PIO2_HI - (double)q * PIO2_LO;
    float r = (float)r_d;

    // Select function and sign based on quadrant
    int qm = q & 3;
    int use_sin = (qm == 1 || qm == 3);
    int sign = (qm == 0 || qm == 3) ? +1 : -1;

    float r2 = r * r;

    if (use_sin) {
        // sin(r) polynomial: r + s3*r^3 + s5*r^5 + s7*r^7 + s9*r^9
        const float s3 = -1.6666667163e-1f;
        const float s5 =  8.3333337680e-3f;
        const float s7 = -1.9841270114e-4f;
        const float s9 =  2.7557314297e-6f;

        float p = ((s9 * r2 + s7) * r2 + s5) * r2 + s3;
        float result = r + r * r2 * p;
        return sign * result;
    } else {
        // cos(r) polynomial: 1 + c2*r2 + c4*r4 + c6*r6 + c8*r8
        const float c2 = -0.5f;
        const float c4 =  4.1666667908e-2f;
        const float c6 = -1.3888889225e-3f;
        const float c8 =  2.4801587642e-5f;

        float p = ((c8 * r2 + c6) * r2 + c4) * r2 + c2;
        float result = 1.0f + r2 * p;
        return sign * result;
    }
}

NOINLINE float aux_tan(float x) {
    // Promote to double for argument reduction (improves precision)
    double xd = (double)x;
    double qd = xd * (double)TWO_OVER_PI;   // how many half-pi multiples
    int q = (int)(qd + (qd >= 0.0 ? 0.5 : -0.5));  // nearest integer

    // Range reduce: r = x - q*(pi/2)
    const double PIO2_HI = 1.57079625129699707031;     // pi/2 high part
    const double PIO2_LO = 7.54978941586159635335e-08; // pi/2 low part
    double r_d = xd - (double)q * PIO2_HI - (double)q * PIO2_LO;
    float r = (float)r_d;

    // For tan: period is pi, so q mod 2 determines sign
    int qm = q & 3;
    int use_cot = (qm == 1 || qm == 3); // tan(x) = ±cot(r) in odd quadrants
    int sign = (qm == 0 || qm == 2) ? +1 : -1;

    // Polynomial approximations
    // sin(r) ≈ r + s3*r^3 + s5*r^5 + s7*r^7 + s9*r^9
    const float s3 = -1.6666667163e-1f;
    const float s5 =  8.3333337680e-3f;
    const float s7 = -1.9841270114e-4f;
    const float s9 =  2.7557314297e-6f;

    // cos(r) ≈ 1 + c2*r^2 + c4*r^4 + c6*r^6 + c8*r^8
    const float c2 = -0.5f;
    const float c4 =  4.1666667908e-2f;
    const float c6 = -1.3888889225e-3f;
    const float c8 =  2.4801587642e-5f;

    float r2 = r * r;
    float sin_r = r + r * r2 * (((s9 * r2 + s7) * r2 + s5) * r2 + s3);
    float cos_r = 1.0f + r2 * (((c8 * r2 + c6) * r2 + c4) * r2 + c2);

    float t;
    if (!use_cot) {
        // tan(r) = sin(r)/cos(r)
        t = sin_r / cos_r;
    } else {
        // cot(r) = cos(r)/sin(r)
        t = cos_r / sin_r;
    }

    // Avoid catastrophic explosion near vertical asymptotes
    // Clip to a large finite value (~1e8)
    if (t > 1e8f)  t = 1e8f;
    if (t < -1e8f) t = -1e8f;

    return sign * t;
}

NOINLINE float aux_atan(float x) {
    const float absx = x < 0 ? -x : x;

    // Coefficients for a rational minimax fit on [0,1]
    const float a0 = 0.9998660f;
    const float a1 = -0.3302995f;
    const float b1 = 0.1801410f;
    const float b2 = -0.0126492f;

    float y;
    if (absx <= 1.0f) {
        float x2 = x * x;
        y = x * (a0 + a1 * x2) / (1.0f + b1 * x2 + b2 * x2 * x2);
    } else {
        float inv = 1.0f / absx;
        float x2 = inv * inv;
        float core = inv * (a0 + a1 * x2) / (1.0f + b1 * x2 + b2 * x2 * x2);
        y = PI_2 - core;
    }

    return x < 0 ? -y : y;
}

NOINLINE float aux_atan2(float y, float x) {
    if (x == 0.0f) {
        if (y > 0.0f)  return  PI_2;
        if (y < 0.0f)  return -PI_2;
        return 0.0f;  // TODO: undefined
    }

    float abs_y = y < 0 ? -y : y;
    float abs_x = x < 0 ? -x : x;
    float angle;

    if (abs_x > abs_y)
        angle = aux_atan(y / x);
    else
        angle = PI_2 - aux_atan(x / y);

    // Quadrant correction
    if (x < 0) angle = (y >= 0) ? angle + PI : angle - PI;
    return angle;
}

NOINLINE float aux_asin(float x) {
    const float PI_2 = 1.57079632679489661923f;
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;

    const float c3 = 0.16666667f;   // ≈ 1/6
    const float c5 = 0.07500000f;   // ≈ 3/40
    const float c7 = 0.04464286f;   // ≈ 5/112

    float x2 = x * x;
    float p = x + x * x2 * (c3 + x2 * (c5 + c7 * x2));
    return p;
}