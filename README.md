# Copapy

Copapy is a python based embedded domain specific language (eDSL) with copy & patch compiler. It uses the python interpreter for compilation. It generates a directed graph of variables and operations. The compiler generates machine code by composing pre-compiled stencils derived from compiled C code.

The Project targets applications that profit from fast implementation (e.g. prototyping) and require low latency realtime execution as well as minimizing risk of implementation errors not caught during compile time. This applies primarily to applications interfacing hardware, where runtime errors might lead to physical damage - for example in the field of robotics, aerospace, embedded systems and control systems in general.

The language aims to be:
- Fast to write & easy to read
- Type safe
- Having a predictable runtime
- No runtime errors

Because the language is an embedded language, it can relay heavily on **python tooling**. While copapy is static typed, it uses Python to derive types during compile time wherever possible. It can get full benefit from python type checkers, to catch type errors even before compilation to improve ergonomics.

## How it works
The **Compilation** step starts with tracing the python code to generate a acyclic directed graph (DAG) of variables and operations. The DAG can be optimized and gets than linearized to a sequence of operations. Each operation gets mapped to a pre-compiled stencil, which is a piece of machine code with placeholders for memory addresses. The compiler generates patch instructions to fill the placeholders with the correct memory addresses. The binary code build from the stencils, data for constants and the patch instructions are than passed to the Runner for execution. The runner allocates memory for the code and data, applies the patch instructions to correct memory addresses and finally executes the code.

## Getting started
To install copapy, you can use pip:

```bash
pip install copapy
```

## Developer Guide
Contributions are welcome, please open an issue or submit a pull request on GitHub.

To get started with developing the package, first clone the repository to your local machine using Git:

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

If the build fails because you have no suitable c compiler installed, you can either install a compiler or use the binary from pypi:

```bash
pip install copapy[dev]
```

When running pytest it will use the binary from pypi but the local python code gets executed from the local repo.

For running all tests you need the stencil object files and the compiled runner. You can download the stencils and binary runner from GitHub or build them with gcc yourself.

For downloading the latest binaries from GitHub run:

```bash
python tools/get_binaries.py
```

To build the binaries from source on Linux run:

```bash
bash tools/build.sh
```

The runner (without the stencils) can be build on windows with:

```
tools\build
```

Ensure that everything is set up correctly by running the tests:

```bash
pytest
```

## License
This project is licensed under GPL - see the [LICENSE](LICENSE) file for details.
