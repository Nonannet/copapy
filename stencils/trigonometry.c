#include "stencil_helper.h"

const float PI      = 3.14159265358979323846f;
const float PI_2    = 1.57079632679489661923f;  // pi/2
const double TWO_OVER_PI = 0.63661977236758134308; // 2/pi
const double PIO2_HI = 1.57079625129699707031;     // pi/2 high part
const double PIO2_LO = 7.54978941586159635335e-08; // pi/2 low part

float aux_sin(float x) {
    return sinf(x);
}

float aux_cos(float x) {
    return cosf(x);
}

float aux_tan(float x) {
    return tanf(x);
}

float aux_atan(float x) {
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

float aux_atan2(float y, float x) {
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

float aux_asin(float x) {
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;

    const float c3 = 0.16666667f;   // ≈ 1/6
    const float c5 = 0.07500000f;   // ≈ 3/40
    const float c7 = 0.04464286f;   // ≈ 5/112

    float x2 = x * x;
    float p = x + x * x2 * (c3 + x2 * (c5 + c7 * x2));
    return p;
}