# Static export example

A copapy program as static part of a C firmware. The program is traced and
compiled by `make` on the build machine, the firmware contains only the
resulting machine code and data: no runner, no patching and no executable RAM
on the target.

The example builds a firmware for a Cortex-M4 with FPU and runs it in QEMU
(machine `mps2-an386`).

## How it works

1. `make` calls [control.py](control.py). It defines the inputs, traces the
   function `control` and calls `cp.export_static` with the addresses the
   program will be located at.
2. `cp.export_static` compiles the program for the target architecture. The
   runner of the build machine patches the code for the given addresses
   (command `LINK_ADDR`) instead of the addresses of its own memory, and the
   patched code is read back.
3. The result is written as `build/gen/control.c` and `build/gen/control.h`:

   ```c
   typedef struct {
       float setpoint;
       float angle;
       float integral;
       int32_t enable;
       float accel[3];
       float samples[8];
   } control_in_t;

   typedef struct { ... } control_out_t;

   int control_init(void);
   void control_step(const control_in_t *in, control_out_t *out);
   ```

   The members of the structs are the names of the `inputs` and `outputs`
   dicts in `control.py`. `control_step` copies the inputs to the data memory
   of the program, runs it and copies the outputs back: no addresses are
   handled in the firmware.
4. The firmware is compiled and linked as usual. The linker script
   [link.ld](link.ld) places two sections at the addresses the program was
   generated for:

   | Section        | Memory | Content                                              |
   |----------------|--------|------------------------------------------------------|
   | `.copapy_code` | Flash  | Machine code, read only                              |
   | `.copapy_data` | RAM    | Constants and variables, initialized by `control_init` |

## Addresses

`COPAPY_CODE_ADDR` and `COPAPY_DATA_ADDR` are defined once in the
[Makefile](Makefile) and passed to `control.py` and to the linker
(`--defsym`). Both must be aligned to 16 bytes. The linker reports an error if
a section overlaps the rest of the firmware, and `control_init` returns -1 if
the program is not located where it was generated for.

## Build and run

Requirements: copapy (`pip install -e .` in the repository root),
`arm-none-eabi-gcc`, `make` and `qemu-system-arm`.

```sh
make run
```

Output:

```
angle=0.0000 torque=2.0050 integral=0.0100 accel_ref=1.0000,0.0000,9.8100 mean=4.5000 ac[0]=-3.5000
angle=0.4010 torque=1.2060 integral=0.0160 accel_ref=0.9207,0.3903,9.8100 mean=4.5000 ac[0]=-3.5000
...
```

## Notes

- Copapy programs have no state. State like the integrator of the example is
  an output that the firmware feeds back as input.
- `control_step` is not reentrant, all calls share one data memory.
- The FPU must be enabled before the first call (see [startup.c](startup.c)).
- The generated code depends on the target architecture (`COPAPY_ARCH`), it
  must match the CPU and FPU of the firmware. `armv7mthumb` is for Cortex-M4
  and M7 with single precision FPU and hard float ABI.
- The generated C source uses GCC attributes (`section`, `aligned`).
