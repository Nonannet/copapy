#include <stdint.h>
#include "stencil_helper.h"

volatile extern int dummy_int;
volatile extern float dummy_float;

NOINLINE float auxsub_get_42(int n) {
    return n * 5.0f + 21.0f;
}

NOINLINE float aux_get_42(float n) {
    return auxsub_get_42(n * 3.0f + 42.0f);
}

// tanh based on expf, with the Taylor series for small arguments where
// 1 - exp(-2x) loses precision
NOINLINE float aux_tanh(float x) {
    float a = fabsf(x);
    if (a < 0.25f) {
        float x2 = x * x;
        return x * (1.0f + x2 * (-1.0f / 3.0f + x2 * (2.0f / 15.0f + x2 * (-17.0f / 315.0f + x2 * (62.0f / 2835.0f)))));
    }
    float e = expf(-2.0f * a);
    float t = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -t : t;
}

#if defined(__tricore__)
/* Math functions of newlib (used for TriCore) set errno, it is
   initialized to be placed in .data, since .bss is not allocated */
static int errno_placeholder = 1;

NOINLINE int *__errno(void) {
    return &errno_placeholder;
}
#endif
