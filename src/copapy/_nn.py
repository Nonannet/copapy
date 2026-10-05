from . import vector
from . import tensor
from . import value
from typing import TypeVar, Any, Sequence, overload
import copapy as cp
from ._arrays import array, _add_array_op
from ._basic_types import ArrayNet

U = TypeVar("U", int, float)
TArr = TypeVar("TArr", array[Any], tensor[Any])


@overload
def relu(x: U) -> U: ...
@overload
def relu(x: value[U]) -> value[U]: ...
@overload
def relu(x: vector[U]) -> vector[U]: ...
@overload
def relu(x: tensor[U]) -> tensor[U]: ...
def relu(x: U | value[U] | vector[U] | tensor[U]) -> Any:
    """Returns x for x > 0 and otherwise 0."""
    ret = x * (x > 0)
    return ret


@overload
def sigmoid(x: U) -> float: ...
@overload
def sigmoid(x: value[U]) -> value[float]: ...
@overload
def sigmoid(x: vector[U]) -> vector[float]: ...
@overload
def sigmoid(x: tensor[U]) -> tensor[float]: ...
def sigmoid(x: U | value[U] | vector[U] | tensor[U]) -> Any:
    """Sigmoid function to map any value to the range (0, 1)."""
    return 1 / (1 + cp.exp(-x))


def _float_array(x: Any, name: str) -> 'array[Any]':
    """Float array from an array, a tensor or nested sequences"""
    if isinstance(x, tensor):
        arr = x._get_array(force=True)
        if arr is None:
            arr = array(list(x.values)).reshape(*x.shape)
    elif isinstance(x, array):
        arr = x
    elif isinstance(x, Sequence):
        arr = array(x)
    else:
        raise TypeError(f"{name} must be an array, a tensor or a nested sequence, not {type(x).__name__}")
    if arr.dtype != 'float':
        arr = arr._binary_op('mul', 1.0)
    return arr


def _pair(x: int | Sequence[int], name: str, minimum: int) -> tuple[int, int]:
    pair = (x, x) if isinstance(x, int) else tuple(x)
    if len(pair) != 2 or not all(isinstance(v, int) and v >= minimum for v in pair):
        raise ValueError(f"{name} must be one or two integers >= {minimum}, got {x}")
    return pair[0], pair[1]


def _conv(x: 'array[float]', weight: 'array[float]', bias: Any, stride: tuple[int, int],
          padding: tuple[int, int], dilation: tuple[int, int]) -> 'array[float]':
    """Convolution of an input [n, ci, h, w] with the weights [co, ci, kh, kw]"""
    n, ci, h, w = x.shape
    co, wci, kh, kw = weight.shape
    if wci != ci:
        raise ValueError(f"Input with {ci} channels does not match weights for {wci} channels")
    oh, ow = ((size + 2 * p - d * (k - 1) - 1) // s + 1
              for size, k, s, p, d in zip((h, w), (kh, kw), stride, padding, dilation))
    if oh < 1 or ow < 1:
        raise ValueError(f"Kernel of shape {weight.shape[2:]} is larger than the padded input of shape {x.shape[2:]}")

    b = array([0.0] * co) if bias is None else _float_array(bias, 'bias')
    if b.shape != (co,):
        raise ValueError(f"Bias of shape {b.shape} does not match {co} output channels")

    params = array([n, ci, h, w, co, kh, kw, oh, ow, *stride, *padding, *dilation], 'int')
    node = _add_array_op('conv2d_floatarr_floatarr', [x.net, weight.net, b.net, params.net], 'float', n * co * oh * ow)
    assert isinstance(node.result, ArrayNet)
    return array._from_net(node.result, (n, co, oh, ow))


def conv2d(x: TArr, weight: Any, bias: Any = None, stride: int | Sequence[int] = 1,
           padding: int | Sequence[int] = 0, dilation: int | Sequence[int] = 1) -> TArr:
    """2D convolution (cross-correlation as in most deep learning frameworks)
    computed by a single array stencil.

    Arguments:
        x: Input of shape (channels, height, width) or (batch, channels, height, width).
        weight: Kernels of shape (out_channels, channels, kernel_height, kernel_width).
        bias: Optional bias of shape (out_channels,).
        stride: Step of the kernel, a single integer or (vertical, horizontal).
        padding: Zero padding on both sides of the input, a single integer
            or (vertical, horizontal).
        dilation: Spacing between the kernel elements, a single integer
            or (vertical, horizontal).

    Returns:
        Float array or tensor (same type as x) of shape (out_channels, out_height, out_width)
        with a leading batch dimension if x has one.
    """
    xa, wa = _float_array(x, 'x'), _float_array(weight, 'weight')
    if xa.ndim not in (3, 4) or wa.ndim != 4:
        raise ValueError(f"conv2d requires a 3D or 4D input and 4D weights, got shapes {xa.shape} and {wa.shape}")
    ret = _conv(xa.reshape(-1, *xa.shape[-3:]), wa, bias,
                _pair(stride, 'stride', 1), _pair(padding, 'padding', 0), _pair(dilation, 'dilation', 1))
    if xa.ndim == 3:
        ret = ret.reshape(*ret.shape[1:])
    return tensor._from_array(ret) if isinstance(x, tensor) else ret


def conv1d(x: TArr, weight: Any, bias: Any = None, stride: int = 1,
           padding: int = 0, dilation: int = 1) -> TArr:
    """1D convolution (cross-correlation as in most deep learning frameworks)
    computed by a single array stencil.

    Arguments:
        x: Input of shape (channels, length) or (batch, channels, length).
        weight: Kernels of shape (out_channels, channels, kernel_length).
        bias: Optional bias of shape (out_channels,).
        stride: Step of the kernel.
        padding: Zero padding on both sides of the input.
        dilation: Spacing between the kernel elements.

    Returns:
        Float array or tensor (same type as x) of shape (out_channels, out_length)
        with a leading batch dimension if x has one.
    """
    xa, wa = _float_array(x, 'x'), _float_array(weight, 'weight')
    if xa.ndim not in (2, 3) or wa.ndim != 3:
        raise ValueError(f"conv1d requires a 2D or 3D input and 3D weights, got shapes {xa.shape} and {wa.shape}")
    ret = _conv(xa.reshape(-1, xa.shape[-2], 1, xa.shape[-1]), wa.reshape(*wa.shape[:2], 1, wa.shape[2]), bias,
                (1, _pair(stride, 'stride', 1)[1]), (0, _pair(padding, 'padding', 0)[1]),
                (1, _pair(dilation, 'dilation', 1)[1]))
    ret = ret.reshape(*((ret.shape[0],) if xa.ndim == 3 else ()), ret.shape[1], ret.shape[3])
    return tensor._from_array(ret) if isinstance(x, tensor) else ret
