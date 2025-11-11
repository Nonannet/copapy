#include <stdint.h>
#include "stencil_helper.h"

//double (*math_pow)(double, double);

volatile extern int dummy_int;
volatile extern float dummy_float;

int floor_div(float arg1, float arg2) {
    float x = arg1 / arg2;
    int i = (int)x;
    if (x < 0 && x != (float)i) i -= 1;
    return i;
}

float aux_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;

    // --- Improved initial guess using bit-level trick ---
    union { float f; uint32_t i; } conv = { x };
    conv.i = (conv.i >> 1) + 0x1fc00000;  // better bias constant
    float y = conv.f;

    // --- Fixed number of Newton-Raphson iterations ---
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);  // 3 fixed iterations

    return y;
}

float aux_get_42(float n) {
    return n + 42.0;
}

float aux_log(float x)
{
    union { float f; uint32_t i; } vx = { x };
    float e = (float)((vx.i >> 23) & 0xFF) - 127.0f;
    vx.i = (vx.i & 0x007FFFFF) | 0x3F800000; // normalized mantissa in [1,2)
    float m = vx.f;

    // 3rd-degree minimax polynomial approximation of log2(m)
    // over [1, 2): log2(m) ≈ p(m) = a*m^3 + b*m^2 + c*m + d
    float p = -0.34484843f * m * m * m + 2.02466578f * m * m - 2.67487759f * m + 1.65149613f;

    float log2x = e + p;
    return log2x * 0.69314718f; // convert log2 → ln
}

float aux_exp(float x)
{
    // Scale by 1/ln(2)
    x = x * 1.44269504089f;
    float xi = (float)((int)x);
    float f = x - xi;

    // Polynomial approximation of 2^f for f ∈ [0,1)
    float p = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f)));

    // Reconstruct exponent
    int ei = (int)xi + 127;
    if (ei <= 0) ei = 0; else if (ei >= 255) ei = 255;
    union { uint32_t i; float f; } v = { (uint32_t)(ei << 23) };

    return v.f * p;
}
