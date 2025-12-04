from . import variable, vector, matrix
import copapy.backend as cpb
from typing import Any, Sequence, overload
import copapy as cp
from ._basic_types import Net, unifloat


@overload
def grad(x: variable[Any], y: variable[Any]) -> unifloat: ...
@overload
def grad(x: variable[Any], y: Sequence[variable[Any]]) -> list[unifloat]: ...
@overload
def grad(x: variable[Any], y: vector[Any]) -> vector[float]: ...
@overload
def grad(x: variable[Any], y: matrix[Any]) -> matrix[float]: ...
def grad(x: variable[Any], y: variable[Any] | Sequence[variable[Any]] | vector[Any] | matrix[float]) -> Any:
    """Returns the partial derivative dx/dy where x needs to be a scalar
    and y might be a scalar, a list of scalars, a vector or matrix.

    Arguments:
        x: Value to return derivative of
        y: Value(s) to derive in respect to

    Returns:
        Derivative of x with the type and dimensions of y
    """
    edges = cpb.get_all_dag_edges([x.source])
    ordered_ops = cpb.stable_toposort(edges)

    net_lookup = {net.source: net for node in ordered_ops for net in node.args}
    grad_dict: dict[Net, unifloat] = dict()

    def add_grad(val: variable[Any], gradient_value: unifloat) -> None:
        grad_dict[val] = grad_dict.get(val, 0.0) + gradient_value

    for node in reversed(ordered_ops):
        print(f"-->   {'x' if node in net_lookup else ' '}", node, f"{net_lookup.get(node)}")
        if node.args:
            args: Sequence[Any] = list(node.args)
            g = 1.0 if node is x.source else grad_dict[net_lookup[node]]
            opn = node.name.split('_')[0]
            a: variable[Any] = args[0]
            b: variable[Any] = args[1] if len(args) > 1 else a

            if opn in ['ge', 'gt', 'eq', 'ne', 'floordiv', 'bwand', 'bwor', 'bwxor']:
                pass  # Derivative is 0 for all ops returning integers

            elif opn == 'add':
                add_grad(a, g)
                add_grad(b, g)

            elif opn == 'sub':
                add_grad(a, g)
                add_grad(b, -g)

            elif opn == 'mul':
                add_grad(a, b * g)
                add_grad(b, a * g)

            elif opn == 'div':
                add_grad(a, g / b)
                add_grad(b, -a * g / (b**2))

            elif opn == 'mod':
                add_grad(a, g)
                add_grad(b, -a * g / b)

            elif opn == 'log':
                add_grad(a, g / a)

            elif opn == 'exp':
                add_grad(a, g * cp.exp(a))

            elif opn == 'pow':
                add_grad(a, (b * (a ** (b - 1))) * g)
                add_grad(b, (a ** b * cp.log(a)) * g)

            elif opn == 'sqrt':
                add_grad(a, g * (0.5 / cp.sqrt(a)))

            #elif opn == 'abs':
            #    add_grad(x, g * cp.sign(x))

            elif opn == 'sin':
                add_grad(a, g * cp.cos(a))

            elif opn == 'cos':
                add_grad(a, g * -cp.sin(a))

            elif opn == 'tan':
                add_grad(a, g * (1 / cp.cos(a) ** 2))

            elif opn == 'asin':
                add_grad(a, g * (1 / cp.sqrt(1 - a**2)))

            elif opn == 'acos':
                add_grad(a, g * (-1 / cp.sqrt(1 - a**2)))

            elif opn == 'atan':
                add_grad(a, g * (1 / (1 + a**2)))

            elif opn == 'atan2':
                denom = a**2 + b**2
                add_grad(a, g * (-b / denom))
                add_grad(b, g * ( a / denom))

            else:
                raise ValueError(f"Operation {opn} not yet supported for auto diff.")

    if isinstance(y, variable):
        return grad_dict[y]
    if isinstance(y, vector):
        return vector(grad_dict[yi] if isinstance(yi, variable) else 0.0 for yi in y)
    if isinstance(y, matrix):
        return matrix((grad_dict[yi] if isinstance(yi, variable) else 0.0 for yi in row) for row in y)
    return [grad_dict[yi] for yi in y]
