"""Copapy program of the static export example.

The program is traced and compiled on the build machine (called by make) and
written as C source for the firmware: control.c and control.h.
"""
import argparse
from typing import Any
import copapy as cp

KP = 2.0     # Proportional gain
KI = 0.5     # Integral gain
DT = 0.01    # Sample time in s
I_MAX = 5.0  # Limit of the integrator
N_SAMPLES = 8


def control(setpoint: Any, angle: Any, integral: Any, enable: Any, accel: Any, samples: Any) -> dict[str, Any]:
    """PI controller with some signal processing. Copapy programs have no
    state: the integrator is an input and an output, the firmware keeps it."""
    error = setpoint - angle
    integral_new = cp.clamp(integral + error * DT, -I_MAX, I_MAX)
    torque = (KP * error + KI * integral_new) * enable

    # Rotate the acceleration about the z axis into the reference frame
    s, c = cp.sin(angle), cp.cos(angle)
    accel_ref = cp.vector([c * accel[0] - s * accel[1],
                           s * accel[0] + c * accel[1],
                           accel[2]])

    # Remove the mean from the samples (array stencils)
    mean = samples.sum() / N_SAMPLES

    return {'torque': torque,
            'integral': integral_new,
            'saturated': cp.abs(integral_new) >= I_MAX,
            'accel_ref': accel_ref,
            'mean': mean,
            'ac': samples - mean}


def make_inputs() -> dict[str, Any]:
    """Inputs of the program with their initial values"""
    return {'setpoint': cp.value(0.0),
            'angle': cp.value(0.0),
            'integral': cp.value(0.0),
            'enable': cp.value(0),
            'accel': cp.vector(cp.value(0.0) for _ in range(3)),
            'samples': cp.array([0.0] * N_SAMPLES)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help="Directory for control.c and control.h")
    parser.add_argument('--arch', required=True, help="Target architecture, e.g. armv7mthumb")
    parser.add_argument('--code-addr', required=True, type=lambda s: int(s, 0), help="Address of the section .copapy_code")
    parser.add_argument('--data-addr', required=True, type=lambda s: int(s, 0), help="Address of the section .copapy_data")
    args = parser.parse_args()

    inputs = make_inputs()
    prog = cp.export_static('control', inputs, control(**inputs), args.out,
                            arch=args.arch, code_addr=args.code_addr, data_addr=args.data_addr)
    print(f"{prog.c_file}: {prog.code_size} bytes code, {prog.data_size} bytes data")
