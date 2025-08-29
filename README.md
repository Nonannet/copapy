# Copapy

## Installation
Installation with pip:
``` bash
pip install copapy
```

## Getting started


## Developer Guide
Contributions are welcome, please open an issue or submit a pull request on GitHub.

To get started with developing the `copapy` package, follow these steps.

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/nonannet/copapy.git
cd copapy
```

It's recommended to setup an venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Install the package and dev-dependencies while keeping the package files
in the current directory:

```bash
pip install -e .[dev]
```

Ensure that everything is set up correctly by running the tests:

```bash
pytest
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
