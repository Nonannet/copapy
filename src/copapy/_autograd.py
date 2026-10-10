from copapy._quaternion import quaternion

from . import value, vector, tensor
import copapy.backend as cpb
from typing import Any, Sequence, overload
import copapy as cp
from ._arrays import array
from ._basic_types import Net, Node, unifloat, ArrayNet, ArrayConst, ArrayOp, ArrayPack, ArrayElement, HeadNode, add_op
from ._linalg import solve_arrays, inv
from ._interp import get_table_delta

# Operations with a derivative of 0 (integer results)
_ZERO_GRAD_OPS = ('ge', 'gt', 'eq', 'ne', 'floordiv', 'bwand', 'bwor', 'bwxor', 'int', 'bool', 'byte', 'sign', 'argsort', 'gtabs')

# Array operations without a derivative rule
_UNSUPPORTED_ARRAY_OPS = ('copy32', 'copy8', 'sort', 'conv2d', 'maxpool2d', 'avgpool2d')


def _resolve(net: Net) -> Net:
    """Net of the scalar stored to an array element, for elements of an array
    packed from scalars"""
    source = net.source
    while isinstance(source, ArrayElement) and isinstance(source.args[0].source, ArrayPack):
        net = source.args[0].source.args[source.index]
        source = net.source
    return net


def _wrap(net: Net) -> Any:
    """Operand of a node as flat array or as value"""
    return array._from_net(net, (net.length,)) if isinstance(net, ArrayNet) else value(net)


def _const_ints(net: Net) -> tuple[int, ...]:
    """Dimensions passed to an array stencil as constant or as constant array"""
    source = net.source
    assert isinstance(source, HeadNode | ArrayConst), "Dimensions of an array operation must be constants"
    return tuple(int(v) for v in (source.values if isinstance(source, ArrayConst) else (source.value,)))


@overload
def grad(x: Any, y: value[Any]) -> unifloat: ...
@overload
def grad(x: Any, y: vector[Any]) -> vector[float]: ...
@overload
def grad(x: Any, y: tensor[Any]) -> tensor[float]: ...
@overload
def grad(x: Any, y: array[Any]) -> array[float]: ...
@overload
def grad(x: Any, y: quaternion) -> quaternion: ...
@overload
def grad(x: Any, y: Sequence[value[Any]]) -> list[unifloat]: ...
def grad(x: Any, y: value[Any] | Sequence[value[Any]] | vector[Any] | tensor[Any] | array[Any] | quaternion) -> Any:
    """Returns the partial derivative dx/dy where x needs to be a scalar
    and y might be a scalar, a list of scalars, a vector, matrix or array. It
    uses automatic differentiation in reverse-mode.

    Element-wise array operations, sum, min, max, dot and matrix products
    are differentiated by array operations. Convolutions, pooling, sorting
    and strided copies (transposing, slicing, broadcasting) are not supported.

    Arguments:
        x: Value to return derivative of
        y: Value(s) to derive in respect to

    Returns:
        Derivative of x with the type and dimensions of y
    """
    assert isinstance(x, value), f"Argument x for grad function must be a copapy value but is {type(x)}."

    y_values: list[Any]
    if isinstance(y, value):
        y_values = [y]
    elif isinstance(y, array):
        y_values = []
    elif isinstance(y, tensor | vector | quaternion):
        y_values = list(y.values)
    else:
        assert isinstance(y, Sequence)
        y_values = list(y)

    def leaf(net: Net) -> Node:
        # The gradient of an array element is taken from the gradient of the array
        source = net.source
        return source.args[0].source if isinstance(source, ArrayElement) else source

    y_nets = {id(v): _resolve(v.net) for v in y_values if isinstance(v, value)}
    leaves = {leaf(net) for net in y_nets.values()} | ({y.net.source} if isinstance(y, array) else set())

    edges = cpb.get_all_dag_edges_between([x.net.source], leaves)
    ordered_ops = cpb.stable_toposort(edges)

    net_lookup = {net.source: net for node in ordered_ops for net in node.args}
    path_nodes = set(ordered_ops)  # Nodes depending on y
    grad_dict: dict[Net, unifloat] = {}  # Gradients of scalars
    array_grads: dict[Net, array[Any]] = {}  # Gradients of arrays as flat arrays
    element_grads: dict[Net, dict[int, unifloat]] = {}  # Gradients of single array elements

    def add_grad(operand: Any, gradient: Any) -> None:
        net: Net = operand.net
        if isinstance(net, ArrayNet):
            if not isinstance(gradient, array):
                # The same gradient for each element
                ones: array[Any] = array([1.0] * net.length)
                gradient = ones * gradient
            if gradient.net.dtype != 'float':
                gradient = gradient._binary_op('mul', 1.0)
            assert gradient.size == net.length, f"Gradient of size {gradient.size} for an array of size {net.length}"
            gradient = gradient.reshape(net.length)
            array_grads[net] = array_grads[net] + gradient if net in array_grads else gradient
        else:
            if isinstance(gradient, array):
                gradient = gradient.sum()  # The scalar is used for each element
            grad_dict[net] = grad_dict.get(net, 0.0) + gradient

    def array_grad(net: ArrayNet) -> 'array[Any] | None':
        """Gradient of an array including the gradients of its single elements"""
        elements = element_grads.pop(net, None)
        if elements:
            packed: array[Any] = array([elements.get(i, 0.0) for i in range(net.length)], 'float')
            array_grads[net] = array_grads[net] + packed if net in array_grads else packed
        return array_grads.get(net)

    for node in reversed(ordered_ops):
        #print(f"-->   {'x' if node in net_lookup else ' '}", node, f"{net_lookup.get(node)}")
        out_net = node.result if isinstance(node, ArrayOp | ArrayPack) else net_lookup.get(node)
        g: Any
        if node is x.net.source:
            g = 1.0
        elif isinstance(out_net, ArrayNet):
            g = array_grad(out_net)
        else:
            g = grad_dict.get(out_net) if out_net is not None else None

        if g is None or not node.args:
            continue  # No derivative (e.g. only used by comparisons) or an input

        if isinstance(node, ArrayElement):
            elements = element_grads.setdefault(node.args[0], {})
            elements[node.index] = elements.get(node.index, 0.0) + g
            continue

        if isinstance(node, ArrayPack):
            for i, arg in enumerate(node.args):
                add_grad(value(arg), g.element(i))
            continue

        parts = node.name.split('_')
        opn = parts[0]
        # Array stencils have additional arguments for the sizes
        args: Sequence[Net] = node.args[:len(parts) - 1] if isinstance(node, ArrayOp) else list(node.args)
        reduction = isinstance(node, ArrayOp) and not isinstance(node.result, ArrayNet)
        a: Any = _wrap(args[0])
        b: Any = _wrap(args[1]) if len(args) > 1 else a

        if opn in _ZERO_GRAD_OPS:
            pass  # Derivative is 0 for all ops returning integers

        elif opn in _UNSUPPORTED_ARRAY_OPS and isinstance(node, ArrayOp):
            raise NotImplementedError(f"Automatic differentiation of the array operation {node.name} is not supported yet")

        elif opn == 'sum':
            add_grad(a, g)

        elif opn == 'sumaxis':
            # Each element of the summed axis gets the gradient of its sum
            m, k, n = _const_ints(node.args[1])
            add_grad(a, g.reshape(m, 1, n).broadcast_to((m, k, n)))

        elif opn == 'dot':
            add_grad(a, b * g)
            add_grad(b, a * g)

        elif opn == 'matvec':
            (m,), (n,) = _const_ints(node.args[2]), _const_ints(node.args[3])
            add_grad(a, g.reshape(m, 1).matmul(b.reshape(1, n)))
            add_grad(b, a.reshape(m, n).T.matmul(g))

        elif opn == 'matmul':
            m, k, n = _const_ints(node.args[2])
            add_grad(a, g.reshape(m, n).matmul(b.reshape(k, n).T))
            add_grad(b, a.reshape(m, k).T.matmul(g.reshape(m, n)))

        elif opn == 'solve':
            # x = a^-1 b: the gradient of b is a^-T g, the gradient of a is -(a^-T g) x^T
            n, r = _const_ints(node.args[2])
            assert isinstance(out_net, ArrayNet)
            sol: array[Any] = array._from_net(out_net, (n, r))
            g_b = solve_arrays(a.reshape(n, n).T, g.reshape(n, r))
            add_grad(b, g_b)
            add_grad(a, -g_b.matmul(sol.T))

        elif opn == 'det':
            # The gradient of det(a) is det(a) a^-T (not defined for a singular matrix)
            (n,) = _const_ints(node.args[1])
            assert out_net is not None
            add_grad(a, inv(a.reshape(n, n).T) * (value(out_net) * g))

        elif opn in ('lerp', 'bsearch'):
            # a is the table (lerp) or the grid (bsearch), b the position
            if args[0].source in path_nodes:
                raise NotImplementedError(f"Automatic differentiation of {opn} to the table is not supported yet")
            if opn == 'lerp':
                # Slope of the table between the two elements around the index
                diff, inside = get_table_delta(a, b)
                add_grad(b, g * (inside * diff))
            else:
                # The index increases by 1 over the interval of the grid around the position
                assert out_net is not None
                diff, inside = get_table_delta(a, _wrap(out_net))
                add_grad(b, g * (inside / diff))

        elif opn in ('mask', 'masknot'):
            # The gradient passes where the value passes
            add_grad(a, add_op(opn, [g if isinstance(g, value) else float(g), b]))

        elif opn in ('min', 'max') and reduction:
            # The derivative is 1 for the elements equal to the result
            assert out_net is not None
            result: value[Any] = value(out_net)
            add_grad(a, (a <= result if opn == 'min' else a >= result) * g)

        elif opn in ('min', 'max'):
            from_a = a < b if opn == 'min' else a > b
            add_grad(a, g * from_a)
            add_grad(b, g * (1 - from_a))

        elif opn == 'add':
            add_grad(a, g)
            add_grad(b, g)

        elif opn == 'sub':
            add_grad(a, g)
            add_grad(b, -g)

        elif opn == 'mul':
            add_grad(a, g * b)
            add_grad(b, g * a)

        elif opn == 'square':
            add_grad(a, g * a * 2)

        elif opn == 'div':
            add_grad(a, g / b)
            add_grad(b, g * (-a) / (b**2))

        elif opn == 'mod':
            add_grad(a, g)
            add_grad(b, g * (-a) / b)

        elif opn == 'log':
            add_grad(a, g / a)

        elif opn == 'exp':
            add_grad(a, g * cp.exp(a))

        elif opn == 'pow':
            add_grad(a, g * cp.pow(a, b - 1) * b)
            add_grad(b, g * cp.pow(a, b) * cp.log(a))

        elif opn == 'sqrt':
            add_grad(a, g * (0.5 / cp.sqrt(a)))

        elif opn == 'abs':
            add_grad(a, g * cp.sign(a))

        elif opn == 'neg':
            add_grad(a, -g)

        elif opn == 'float':
            add_grad(a, g)  # int to float conversion

        elif opn == 'sin':
            add_grad(a, g * cp.cos(a))

        elif opn == 'cos':
            add_grad(a, g * -cp.sin(a))

        elif opn == 'tan':
            add_grad(a, g * (1 / cp.cos(a) ** 2))

        elif opn == 'tanh':
            add_grad(a, g * (1 - cp.tanh(a) ** 2))

        elif opn == 'asin':
            add_grad(a, g * (1 / cp.sqrt(1 - a**2)))

        elif opn == 'acos':
            add_grad(a, g * (-1 / cp.sqrt(1 - a**2)))

        elif opn == 'atan':
            add_grad(a, g * (1 / (1 + a**2)))

        elif opn == 'atan2':
            denom = a**2 + b**2
            add_grad(a, g * b / denom)
            add_grad(b, g * (-a) / denom)

        else:
            raise ValueError(f"Operation {opn} not yet supported for auto diff.")

    def result_of(v: Any) -> Any:
        if not isinstance(v, value):
            return 0.0
        net = y_nets[id(v)]
        source = net.source
        if isinstance(source, ArrayElement):
            arr = array_grads.get(source.args[0])
            return arr.element(source.index) if arr is not None else 0.0
        return grad_dict.get(net, 0.0)

    if isinstance(y, value):
        return result_of(y)
    if isinstance(y, array):
        ret = array_grads.get(y.net)
        if ret is None:
            ret = array([0.0] * y.size)
        return ret.reshape(*y.shape)
    if isinstance(y, vector):
        return vector(result_of(yi) for yi in y.values)
    if isinstance(y, quaternion):
        return quaternion(result_of(yi) for yi in y.values)
    if isinstance(y, tensor):
        return tensor([result_of(yi) for yi in y.values], y.shape)
    return [result_of(yi) for yi in y_values]
