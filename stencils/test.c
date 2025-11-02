#include "aux_functions.c"
#include "trigonometry.c"

int main() {
    // Test aux functions
    float a = 16.0f;
    float sqrt_a = aux_sqrt(100000.0f);
    float pow_a = fast_pow_float(a, 0.5f);
    float div_result = (float)floor_div(-7.0f, 3.0f);
    float sin_30 = aux_sin(30.0f);
    float cos_60 = aux_cos(60.0f);
    float tan_45 = aux_tan(45.0f);
    float g42 = aux_get_42(0.0f);
    return 0;
}
