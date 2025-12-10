# Copapy
Copapy is a Python framework for deterministic, low-latency realtime computation, targeting hardware applications - for example in the fields of robotics, aerospace, embedded systems and control systems in general.

GPU frameworks like PyTorch, JAX and TensorFlow jump-started the development in the field of AI. With the right balance of flexibility and performance, they allow for fast iteration of new ideas while still being performant enough to test or even use them in production.

This is exactly what Copapy aims for - but in the field of embedded realtime computation. While making use of the ergonomics of Python, the tooling, and the general Python ecosystem, Copapy runs seamlessly optimized machine code. Despite being highly portable, the **copy-and-patch** compiler allows for effortless and fast deployment without any dependencies beyond Python. It's designed to feel like writing Python scripts with a shallow learning curve, but under the hood it produces high-performance, statically typed and memory-safe code with a minimized set of possible runtime errors[^1]. To maximize productivity, the framework provides detailed type hints to catch most errors even before compilation.

Embedded systems come with a variety of CPU architectures. The **copy-and-patch** compiler already supports the most common ones[^3], and porting it to new architectures is straightforward if a C compiler for the target architecture is available[^2]. The generated code depends only on the CPU architecture. The generated binaries neither perform system calls nor rely on external libraries like libc. This makes Copapy both highly deterministic and easy to deploy on different realtime operating systems (RTOS) or bare metal.

The main features can be summarized as:
- Fast to write & easy to read
- Memory and type safety with a minimal set of runtime errors
- Deterministic execution
- Autograd for efficient realtime optimization
- Optimized machine code for x86_64, AArch64 and ARMv7
- Highly portable to new architectures
- Small Python package with minimal dependencies and no cross-compile toolchain required

Execution of the compiled code is managed by a runner application. The runner is implemented in C and handles I/O and communication with the Copapy framework. The overall design emphasizes minimal complexity to simplify portability, since this part must be adapted for the individual hardware/applications. Because patching of memory addresses is done by the runner, the different architecture-specific relocation types are unified to an architecture-independent format by Copapy before sending the patch instructions to the runner. This keeps the runner implementation simple.

![Copapy architecture](docs/source/media/copapy.svg)

The design targets either an architecture with a realtime-patched Linux kernel - where the runner uses the same CPU and memory as Linux but executes in a realtime thread - or a setup where even higher determinism is required. In such cases, the runner can be executed on a separate crossover MCU running on bare metal or a RTOS.

The Copapy framework also includes a Python-based runner module. This allows frictionless testing of code and might be valuable for using Copapy in conventional application development.

## Current state
While hardware I/O is obviously a core aspect of the project, it is not yet available. Therefore, this package is currently a proof of concept with limited direct use. However, the computation engine is fully functional and available for testing and experimentation simply by installing the package. The project is now close to being ready for integration into its first demonstration hardware platform.

Currently in development:
- Array stencils for handling very large arrays and generating SIMD-optimized code - e.g., for machine vision and neural network applications
- Support for Thumb instructions required by ARM*-M targets (for crossover MCUs)
- Constant regrouping for further symbolic optimization of the computation graph

## Install
To install Copapy, you can use pip. Precompiled wheels are available for Linux (x86_64, AArch64, ARMv7), Windows (x86_64) and macOS (x86_64, AArch64):

```bash
pip install copapy
```

## Examples

### Basic example

A very simple example program using Copapy can look like this:

```python
import copapy as cp

# Define variables
a = cp.value(0.25)
b = cp.value(0.87)

# Define computations
c = a + b * 2.0
d = c ** 2 + cp.sin(a)
e = cp.sqrt(b)

# Create a target (default is local), compile and run
tg = cp.Target()
tg.compile(c, d, e)
tg.run()

# Read the results
print("Result c:", tg.read_value(c))
print("Result d:", tg.read_value(d))
print("Result e:", tg.read_value(e))
```

### Inverse kinematics

Another example using autograd in Copapy, here implementing
gradient descent to solve an inverse kinematics problem for
a two-joint 2D arm:

```python
import copapy as cp

# Arm lengths
l1, l2 = 1.8, 2.0

# Target position
target = cp.vector([0.7, 0.7])

# Learning rate for iterative adjustment
alpha = 0.1

def forward_kinematics(theta1, theta2):
    """Return positions of joint and end-effector."""
    joint = cp.vector([l1 * cp.cos(theta1), l1 * cp.sin(theta1)])
    end_effector = joint + cp.vector([l2 * cp.cos(theta1 + theta2),
                                     l2 * cp.sin(theta1 + theta2)])
    return joint, end_effector

# Start values
theta = cp.vector([cp.value(0.0), cp.value(0.0)])

# Iterative inverse kinematics
for _ in range(48):
    joint, effector = forward_kinematics(theta[0], theta[1])
    error = ((target - effector) ** 2).sum()

    theta -= alpha * cp.grad(error, theta)

tg = cp.Target()
tg.compile(error, theta, joint)
tg.run()

print(f"Joint angles: {tg.read_value(theta)}")
print(f"Joint position: {tg.read_value(joint)}")
print(f"End-effector position: {tg.read_value(effector)}")
print(f"quadratic error = {tg.read_value(error)}")
```

```
Joint angles: [-0.7221821546554565, 2.6245293617248535]
Joint position: [1.3509329557418823, -1.189529299736023]
End-effector position: [0.6995794177055359, 0.7014330625534058]
quadratic error = 2.2305819129542215e-06
```

## How it works

The compilation step starts with tracing the Python code to generate an acyclic directed graph (DAG) of variables and operations. The code can contain functions, closures, branching, and so on, but conditional branching is only allowed when the condition is known at tracing time (a `cp.iif` function exists to work around this). In the next step, this DAG is optimized and linearized into a sequence of operations. Each operation is mapped to a precompiled stencil or a combination of several stencils. A stencil is a piece of machine code with placeholders for memory addresses pointing to other code or data. The compiler generates patch instructions that fill these placeholders with the correct memory addresses.

After compilation, the binary code built from the stencils, the constant data, and the patch instructions is handed to the runner for execution. The runner allocates memory for code and data, copies both into place, applies the patch instructions, and finally executes the code.

The C code for a very simple stencil can look like this:

```c
add_float_float(float arg1, float arg2) {
    result_float_float(arg1 + arg2, arg2);
}
```

The call to the dummy function `result_float_float` ensures that the compiler keeps the result and the second operand in registers for later use. The dummy function acts as a placeholder for the next stencil. Copapy uses two virtual registers, which map on most relevant architectures to actual hardware registers. Data that cannot be kept in a register is stored in statically allocated heap memory. Stack memory may be used inside some stencils, but its usage is essentially fixed and independent of the Copapy program, so total memory requirements are known at compile time.

The machine code for the function above, compiled for x86_64, looks like this:

```assembly
0000000000000000 <add_float_float>:
   0:    f3 0f 58 c1              addss  %xmm1,%xmm0
   4:    e9 00 00 00 00           jmp    9 <.LC1+0x1>
            5: R_X86_64_PLT32    result_float_float-0x4
```

Based on the relocation entry for the `jmp` to the symbol `result_float_float`, the `jmp` instruction is stripped when it is the last instruction in a stencil. Thus, a Copapy addition operation results in a single instruction. For stencils containing multiple branch exits, only the final `jmp` is removed; the others are patched to jump to the next stencil.

For more complex operations - where inlining is less useful - stencils call a non-stencil function, such as in this example:

```assembly
0000000000000000 <sin_float>:
   0:    48 83 ec 08              sub    $0x8,%rsp
   4:    e8 00 00 00 00           call   9 <sin_float+0x9>
            5: R_X86_64_PLT32    sinf-0x4
   9:    48 83 c4 08              add    $0x8,%rsp
   d:    e9 00 00 00 00           jmp    12 <.LC0+0x2>
            e: R_X86_64_PLT32    result_float-0x4
```

Unlike stencils, non-stencil functions are not stripped and do not need to be tail-call-optimizable.

Non-stencil functions and constants are stored together with the stencils in an ELF object file for each supported CPU architecture. The required non-stencil functions and constants are bundled during compilation. The compiler includes only the data and code required for the specific program.

The whole compilation process is independent of the actual instruction set. It relies purely on relocation entries and symbol metadata from the ELF file generated by the C compiler.

## Developer Guide

Feedback and contributions are welcome - please open an issue or submit a pull request on GitHub.

To get started with development, first clone the repository:

```bash
git clone https://github.com/Nonannet/copapy.git
cd copapy
```

You may set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: `.venv\Scripts\activate`
```

Build and install the package and dev dependencies:

```bash
pip install -e .[dev]
```

If the build fails because no suitable C compiler is installed, you can either install one or use the binary package from PyPI:

```bash
pip install copapy[dev]
```

When running pytest, it will use the binary components from PyPI, but all Python code is executed from the local repository.

To run all tests, you need the stencil object files and the compiled runner. You can download them from GitHub or build them yourself with gcc.

Download the latest binaries from GitHub:

```bash
python tools/get_binaries.py
```

Build the binaries from source on Linux:

```bash
bash tools/build.sh
```

Run the tests:

```bash
pytest
```

## License

This project is licensed under GPL - see the [LICENSE](LICENSE) file for details.

[^1]: Errors like divide-by-zero are currently still possible. The feasibility of tracking value ranges in the type system is under investigation to enable compile-time checks.

[^2]: The compiler must support tail-call optimization (TCO). Currently, GCC is supported. Porting to a new architecture requires implementing a subset of relocation types used by that architecture.

[^3]: Supported architectures: x86_64, AArch64, ARMv7 (non-Thumb). ARMv6/7-M (Thumb) support is in development. Code for x86 32-bit exists but has unresolved issues and a low priority.
