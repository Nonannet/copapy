#include <stdint.h>
#include "stencil_helper.h"

volatile extern int dummy_int;
volatile extern float dummy_float;

int floor_div(float arg1, float arg2) {
    float x = arg1 / arg2;
    int i = (int)x;
    if (x < 0 && x != (float)i) i -= 1;
    return i;
}

float aux_get_42(float n) {
    return n + 42.0;
}
