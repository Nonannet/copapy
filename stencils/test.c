#include <stdio.h>
#include "aux_functions.c"

int main() {
    // Test aux functions
    int errors = 0;

    float g42 = aux_get_42(0.0f);
    if (g42 != 231.0f) {
        printf("aux_get_42(0) = %f, expected 231\n", g42);
        errors++;
    }

    // aux_tanh against tanhf of the C library, including the range of the Taylor series
    for (int i = -4000; i <= 4000; i++) {
        float x = (float)i * 0.0025f;
        float res = aux_tanh(x);
        float ref = tanhf(x);
        if (fabsf(res - ref) > 1e-6f) {
            printf("aux_tanh(%f) = %.9f, expected %.9f\n", x, res, ref);
            errors++;
        }
    }

    if (errors) {
        printf("%d errors\n", errors);
        return 1;
    }
    printf("aux functions ok\n");
    return 0;
}
