"""
Copapy is a Python framework for deterministic, low-latency
realtime computation with automatic differentiation, targeting
hardware applications - for example in the fields of robotics,
aerospace, embedded systems and control systems in general.

Main features:
- Automatic differentiation (reverse-mode)
- Generates optimized machine code
- Highly portable to new architectures
- Small Python package with minimal dependencies

Example usage:
    >>> import copapy as cp

    >>> # Define variables
    >>> a = cp.value(0.25)
    >>> b = cp.value(0.87)

    >>> # Define computations
    >>> c = a + b * 2.0
    >>> d = c ** 2 + cp.sin(a)
    >>> e = cp.sqrt(b)

    >>> # Create a target (default is local), compile and run
    >>> tg = cp.Target()
    >>> tg.compile(c, d, e)
    >>> tg.run()

    >>> # Read the results
    >>> print("Result c:", tg.read_value(c))
    >>> print("Result d:", tg.read_value(d))
    >>> print("Result e:", tg.read_value(e))
"""

from ._target import Target, jit
from ._basic_types import NumLike, value, generic_sdb, iif
from ._vectors import vector, distance, scalar_projection, angle_between, rotate_vector, vector_projection
from ._matrices import matrix, identity, zeros, ones, diagonal, eye
from ._math import sqrt, abs, sign, sin, cos, tan, asin, acos, atan, atan2, log, exp, pow, get_42, clamp, min, max, relu
from ._autograd import grad

__all__ = [
    "Target",
    "NumLike",
    "value",
    "generic_sdb",
    "iif",
    "vector",
    "matrix",
    "identity",
    "zeros",
    "ones",
    "diagonal",
    "sqrt",
    "abs",
    "sin",
    "sign",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "log",
    "exp",
    "pow",
    "get_42",
    "clamp",
    "min",
    "max",
    "relu",
    "distance",
    "scalar_projection",
    "angle_between",
    "rotate_vector",
    "vector_projection",
    "grad",
    "eye",
    "jit"
]
