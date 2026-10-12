/* Firmware of the static export example: runs the copapy program
   generated from control.py in a control loop with a simple plant model. */
#include <stdint.h>
#include "control.h"
#include "semihost.h"

/* Prints a number with 4 decimal places */
static void print_float(float number) {
    char text[16];
    char *pos = text + sizeof(text);
    int negative = number < 0;
    uint32_t scaled = (uint32_t)((negative ? -number : number) * 10000.0f + 0.5f);

    *--pos = 0;
    for (int digit = 0; digit < 4 || scaled; digit++) {
        *--pos = (char)('0' + scaled % 10);
        scaled /= 10;
        if (digit == 3) {
            *--pos = '.';
            if (!scaled) *--pos = '0';
        }
    }
    if (negative) *--pos = '-';
    semihost_write(pos);
}

static void print_value(const char *name, float number) {
    semihost_write(name);
    print_float(number);
}

int main(void) {
    if (control_init() != 0) {
        semihost_write("The copapy program is not located at the address it was generated for\n");
        return 1;
    }

    control_in_t in = {
        .setpoint = 1.0f,
        .enable = 1,
        .accel = {1.0f, 0.0f, 9.81f},
        .samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
    };
    control_out_t out;

    for (int step = 0; step < 5; step++) {
        control_step(&in, &out);

        print_value("angle=", in.angle);
        print_value(" torque=", out.torque);
        print_value(" integral=", out.integral);
        print_value(" accel_ref=", out.accel_ref[0]);
        print_value(",", out.accel_ref[1]);
        print_value(",", out.accel_ref[2]);
        print_value(" mean=", out.mean);
        print_value(" ac[0]=", out.ac[0]);
        semihost_write(out.saturated ? " saturated\n" : "\n");

        /* The firmware keeps the state of the controller */
        in.integral = out.integral;

        /* Plant model: the angle follows the torque */
        in.angle += 0.2f * out.torque;
    }

    return 0;
}
