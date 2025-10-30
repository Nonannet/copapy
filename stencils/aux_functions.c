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

__attribute__((noinline)) float aux_sqrt(float n) {
    if (n < 0) return -1;

    float x = n;             // initial guess
    float epsilon = 0.00001; // desired accuracy

    while ((x - n / x) > epsilon || (x - n / x) < -epsilon) {
        x = 0.5 * (x + n / x);
    }

    return x;
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
