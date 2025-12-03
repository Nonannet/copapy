from . import variable, vector
import copapy.backend as cpb
from typing import Any, Sequence, overload
import copapy as cp
from ._basic_types import Net, unifloat


@overload
def grad(var: variable[Any], to: variable[Any]) -> unifloat: ...
@overload
def grad(var: variable[Any], to: Sequence[variable[Any]]) -> Sequence[unifloat]: ...
@overload
def grad(var: variable[Any], to: vector[Any]) -> vector[float]: ...
def grad(var: variable[Any], to: variable[Any] | Sequence[variable[Any]] | vector[Any]) -> unifloat | Sequence[unifloat] | vector[float]:
    edges = cpb.get_all_dag_edges([var.source])
    ordered_ops = cpb.stable_toposort(edges)

    net_lookup = {net.source: net for node in ordered_ops for net in node.args}
    grad_dict: dict[Net, unifloat] = dict()

    def add_grad(val: variable[Any], gradient_value: unifloat) -> None:
        grad_dict[val] = grad_dict.get(val, 0.0) + gradient_value

    for node in reversed(ordered_ops):
        print(f"-->   {'x' if node in net_lookup else ' '}", node, f"{net_lookup.get(node)}")
        if node.args:
            args: Sequence[Any] = [v for v in node.args]
            g = 1.0 if node is var.source else grad_dict[net_lookup[node]]
            opn = node.name.split('_')[0]
            x: variable[Any] = args[0] 
            y: variable[Any] = args[1] if len(args) > 1 else x

            if opn in ['ge', 'gt', 'eq', 'ne']:
                pass  # Derivative is 0

            elif opn == 'add':
                add_grad(x, g)
                add_grad(y, g)

            elif opn == 'sub':
                add_grad(x, g)
                add_grad(y, -g)

            elif opn == 'mul':
                add_grad(x, y * g)
                add_grad(y, x * g)

            elif opn == 'div':  
                add_grad(x, g / y)
                add_grad(y, -x * g / (y**2))

            elif opn == 'pow':
                add_grad(x, (y * (x ** (y - 1))) * g)
                add_grad(y, (x ** y * cp.log(x)) * g)

            elif opn == 'sqrt':
                add_grad(x, g * (0.5 / cp.sqrt(x)))

            elif opn == 'abs':
                add_grad(x, g * cp.sign(x))

            elif opn == 'sin':
                add_grad(x, g * cp.cos(x))

            elif opn == 'cos':
                add_grad(x, g * -cp.sin(x))

            elif opn == 'tan':
                add_grad(x, g * (1 / cp.cos(x) ** 2))

            elif opn == 'asin':
                add_grad(x, g * (1 / cp.sqrt(1 - x**2)))

            elif opn == 'acos':
                add_grad(x, g * (-1 / cp.sqrt(1 - x**2)))

            elif opn == 'atan':
                add_grad(x, g * (1 / (1 + x**2)))

            elif opn == 'atan2':
                denom = x**2 + y**2
                add_grad(x, g * (-y / denom))
                add_grad(y, g * ( x / denom))

            elif opn == 'log':
                add_grad(x, g / x)

            elif opn == 'exp':
                add_grad(x, g * cp.exp(x))

            elif opn == 'gt':
                add_grad(x, g)
                add_grad(y, -g)

            else:
                raise ValueError(f"Operation {opn} not yet supported for auto diff.")

    if isinstance(to, variable):
        return grad_dict[to]
    if isinstance(to, vector):
        return vector(grad_dict[dvar] for dvar in to)
    return [grad_dict[dvar] for dvar in to]