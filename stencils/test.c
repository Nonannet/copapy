#include "aux_functions.c"
#include "trigonometry.c"

int main() {
    // Test aux functions
    float a = 16.0f;
    float sqrt_a = aux_sqrt(100000.0f);
    float div_result = (float)floor_div(-7.0f, 3.0f);
    float sin_30 = aux_sin(30.0f);
    float cos_60 = aux_cos(60.0f);
    float tan_45 = aux_tan(45.0f);
    float atan_15 = aux_atan(1.5f);
    float asin_15 = aux_asin(1.5f);
    float atan2_15 = aux_atan2(1.5f, 1.5f);
    float exp_5 = aux_exp(5.0);
    float log_5 = aux_log(5.0);
    float g42 = aux_get_42(0.0f);
    return 0;
}
