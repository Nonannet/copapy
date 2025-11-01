#include <stdint.h>

//double (*math_pow)(double, double);

volatile extern int dummy_int;
volatile extern float dummy_float;

__attribute__((noinline)) int floor_div(float arg1, float arg2) {
    float x = arg1 / arg2;
    int i = (int)x;
    if (x < 0 && x != (float)i) i -= 1;
    return i;
}

__attribute__((noinline)) float aux_sqrt(float x) {
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

__attribute__((noinline)) float aux_get_42(float n) {
    return n + 42.0;
}

float fast_pow_float(float base, float exponent) {
    union {
        float f;
        uint32_t i;
    } u;

    u.f = base;
    int32_t x = u.i;
    int32_t y = (int32_t)(exponent * (x - 1072632447) + 1072632447);
    u.i = (uint32_t)y;
    return u.f;
}
