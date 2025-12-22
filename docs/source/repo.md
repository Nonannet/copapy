# Code

Primary code repository is on GitHub: [github.com/Nonannet/copapy](https://github.com/Nonannet/copapy).

[Issues](https://github.com/Nonannet/copapy/issues) and [pull requests](https://github.com/Nonannet/copapy/pulls) can be created there.

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


