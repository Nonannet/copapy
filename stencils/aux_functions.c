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

#if defined(__tricore__)
/* Math functions of newlib (used for TriCore) set errno, it is
   initialized to be placed in .data, since .bss is not allocated */
static int errno_placeholder = 1;

NOINLINE int *__errno(void) {
    return &errno_placeholder;
}
#endif
