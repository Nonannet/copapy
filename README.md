# Copapy
Copapy is a python framework for deterministic low latency realtime computations, targeting hardware applications - for example in the field of robotics, aerospace, embedded systems and control systems in general.

GPU frameworks like PyTorch, JAX and TensorFlow jump started the development in the field of AI. With the right balance of flexibility and performance they allow for fast iterations of new ideas while being performant enough to test them or even use them in production.

This is exactly what Copapy is aiming for - but in the field of embedded realtime computation. While making use of the ergonomics of Python, the tooling and the general Python ecosystem, Copapy runs seamlessly optimized machine code. Despite being highly portable, the **copy and patch** compiler allows for effortless and fast deployment, without any dependencies beyond Python. It's designed to feel like writing python scripts, with a flat learning curve. But under the hood it produces high performance static typed and memory save code with a minimized set of possible runtime errors[^1]. To maximize productivity the framework provides detailed type hints to catch most errors even before compilation.

Embedded systems comes with a variety of CPU architectures. The **copy and patch** compiler already supports the most common ones [^3] and porting it to new architectures is effortless if a C compiler for the target architecture is available [^2]. The generated code depends only on the CPU architecture. The actual generated code does neither do system calls nor calling external libraries like libc. This allows Copapy for one to be highly deterministic and for the other it makes targeting different realtime operating systems or bare metal straight forward. 

The summarized main features are:
- Fast to write & easy to read
- Memory and type safety, minimal set of runtime errors [^1]
- deterministic execution
- Auto grad for efficient realtime optimizations
- Optimized machine code for the target architectures x68_64, Aarch64 and ARMv7 [^3]
- Very portable to new architectures [^2]
- Small python package, minimal dependencies, no cross compile toolchain required

## Current state
While obviously hardware IO is a core aspect, this is not yet available. Therefore this package is at the moment a proof of concept with limited direct use. However the computation part is fully working and available for testing and playing with it by simply installing the package. At this point the project is quite close to being ready for integration into the first demonstration hardware platform.

Currently worked on:
- Array stencils for handling very large arrays and generate SIMD optimized code - e.g. for machine vision and neural network applications.
- For targeting Crossover‑MCUs, support for Thumb instructions required by ARM*-M is on the way.
- Constant-regrouping for symbolic optimization of the computation graph.

## Getting started & example
To install copapy, you can use pip. Precompiled wheels are available for Linux (x86_64, Aarch64 and ARMv7), Windows (x86_64) and Mac OS (x86_64, Aarch64):

```bash
pip install copapy
```

A very simple example program using copapy can look like this:

```python
import copapy as cp

# Define variables
a = cp.variable(0.25)
b = cp.variable(0.87)

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

An other example using autograd in copapy. Here for for implementing
gradient descent to solve a reverse kinematic problem for
a two joint 2D arm:

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
theta = cp.vector([cp.variable(0.0), cp.variable(0.0)])

# Iterative inverse kinematic
for _ in range(48):
    joint, effector = forward_kinematics(theta[0], theta[1])
    error = ((target - effector) ** 2).sum()

    grad_vec = cp.grad(error, theta)
    theta -= alpha * grad_vec

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
The **Compilation** step starts with tracing the python code to generate an acyclic directed graph (DAG) of variables and operations. The DAG can be optimized and gets than linearized to a sequence of operations. Each operation gets mapped to a pre-compiled stencil, which is a piece of machine code with placeholders for memory addresses. The compiler generates patch instructions to fill the placeholders with the correct memory addresses. The binary code build from the stencils, data for constants and the patch instructions are than passed to the runner for execution. The runner allocates memory for the code and data, applies the patch instructions to correct memory addresses and finally executes the code.

## Developer Guide
Contributions are welcome, please open an issue or submit a pull request on GitHub.

To get started with developing the package, first clone the repository using Git:

```bash
git clone https://github.com/Nonannet/copapy.git
cd copapy
```

You may setup a venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows `.venv\Scripts\activate`
```

Build and install the package and dev dependencies:

```bash
pip install -e .[dev]
```

If the build fails because you have no suitable c compiler installed, you can either install a compiler (obviously) or use the binary from pypi:

```bash
pip install copapy[dev]
```

When running pytest it will use the binary part from pypi but all the python code gets executed from the local repo.

For running all tests you need the stencil object files and the compiled runner. You can download the stencils and binary runner from GitHub or build them with gcc yourself.

For downloading the latest binaries from GitHub run:

```bash
python tools/get_binaries.py
```

To build the binaries from source on Linux run:

```bash
bash tools/build.sh
```

Ensure that everything is set up correctly by running the tests:

```bash
pytest
```

## License
This project is licensed under GPL - see the [LICENSE](LICENSE) file for details.

[^1]: Currently errors like divide by zero are possible. The feasibility of tacking value ranges in the type system is under investigation to be able to do checks at compile time.
[^2]: The compiler must support TCO (tail call optimization). Currently gcc as C compiler is supported. Porting to a new architecture requires to implement a subset of relocation types used by the architecture.
[^3]: Supported are x68_64, Aarch64, ARMv7 (non-Thumb); ARMv6/7-M (Thumb) is under development; code for x68 32 Bit is present but has open issues (low priority).