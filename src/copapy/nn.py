from . import vector
from . import tensor
from . import value
from typing import TypeVar, Any, Sequence, overload
import copapy as cp
from ._arrays import array, ArrayType, _add_array_op, _size
from ._basic_types import ArrayNet

__all__ = ["relu", "sigmoid", "softmax", "linear", "conv1d", "conv2d",
           "max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d", "global_avg_pool2d"]

U = TypeVar("U", int, float)
TArr = TypeVar("TArr", array[Any], tensor[Any])
TVec = TypeVar("TVec", array[Any], tensor[Any], vector[Any])


@overload
def relu(x: U) -> U: ...
@overload
def relu(x: value[U]) -> value[U]: ...
@overload
def relu(x: vector[U]) -> vector[U]: ...
@overload
def relu(x: tensor[U]) -> tensor[U]: ...
@overload
def relu(x: array[U]) -> array[U]: ...
def relu(x: U | value[U] | array[U] | vector[U] | tensor[U]) -> Any:
    """Returns x for x > 0 and otherwise 0."""
    if isinstance(x, array):
        return x._binary_op('max', 0)
    if isinstance(x, ArrayType):
        packed = x._try_array_op(0, 'max')
        if packed is not None:
            return packed
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
@overload
def sigmoid(x: array[U]) -> array[float]: ...
def sigmoid(x: Any) -> Any:
    """Sigmoid function to map any value to the range (0, 1)."""
    return 1 / (1 + cp.exp(-x))


def softmax(x: TVec) -> TVec:
    """Softmax function: maps the elements to the range (0, 1) with a sum of 1,
    larger elements get a larger share (probabilities of a classification).

    Arguments:
        x: 1D vector, tensor or array

    Returns:
        Float vector, tensor or array (same type as x) of the same length.
    """
    if not isinstance(x, ArrayType | array):
        raise TypeError(f"softmax requires a vector, tensor or array, not {type(x).__name__}")
    if x.ndim != 1:
        raise ValueError(f"softmax requires a 1D input, got shape {x.shape}")
    e: Any = cp.exp(x - cp.max(x))  # The largest exponent is 0: no overflow
    ret: TVec = e / e.sum()
    return ret


def _float_array(x: Any, name: str) -> 'array[Any]':
    """Float array from an array, a tensor, a vector or nested sequences"""
    if isinstance(x, ArrayType):
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
          padding: tuple[int, int], dilation: tuple[int, int],
          out_size: tuple[int, int] | None = None) -> 'array[float]':
    """Convolution of an input [n, ci, h, w] with the weights [co, ci, kh, kw]. The padding
    is applied in front of the input, the output size is given by out_size or by the same
    padding behind the input."""
    n, ci, h, w = x.shape
    co, wci, kh, kw = weight.shape
    if wci != ci:
        raise ValueError(f"Input with {ci} channels does not match weights for {wci} channels")
    if out_size is None:
        oh, ow = ((size + 2 * p - d * (k - 1) - 1) // s + 1
                  for size, k, s, p, d in zip((h, w), (kh, kw), stride, padding, dilation))
    else:
        oh, ow = out_size
    if oh < 1 or ow < 1:
        raise ValueError(f"Kernel of shape {weight.shape[2:]} is larger than the padded input of shape {x.shape[2:]}")

    b = array([0.0] * co) if bias is None else _float_array(bias, 'bias')
    if b.shape != (co,):
        raise ValueError(f"Bias of shape {b.shape} does not match {co} output channels")

    params = array([n, ci, h, w, co, kh, kw, oh, ow, *stride, *padding, *dilation], 'int')
    node = _add_array_op('conv2d_floatarr_floatarr', [x.net, weight.net, b.net, params.net], n * co * oh * ow)
    assert isinstance(node.result, ArrayNet)
    return array._from_net(node.result, (n, co, oh, ow))


def linear(x: TVec, weight: Any, bias: Any = None) -> TVec:
    """Fully connected layer: weight @ x + bias

    Arguments:
        x: Input of shape (in_features,) or (batch, in_features).
        weight: Weights of shape (out_features, in_features).
        bias: Optional bias of shape (out_features,).

    Returns:
        Float array, tensor or vector (same type as x) of shape (out_features,)
        with a leading batch dimension if x has one.
    """
    xa, wa = _float_array(x, 'x'), _float_array(weight, 'weight')
    if xa.ndim not in (1, 2) or wa.ndim != 2:
        raise ValueError(f"linear requires a 1D or 2D input and 2D weights, got shapes {xa.shape} and {wa.shape}")
    if xa.shape[-1] != wa.shape[1]:
        raise ValueError(f"Input with {xa.shape[-1]} features does not match weights for {wa.shape[1]} features")
    ret: Any = wa.matmul(xa.T if xa.ndim == 2 else xa)
    if xa.ndim == 2:
        ret = ret.T  # (out_features, batch) to (batch, out_features)
    if bias is not None:
        b = _float_array(bias, 'bias')
        if b.shape != (wa.shape[0],):
            raise ValueError(f"Bias of shape {b.shape} does not match {wa.shape[0]} output features")
        ret = ret + b
    out: TVec = type(x)._from_array(ret) if isinstance(x, ArrayType) else ret
    return out


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


def _pool(x: Any, op: str, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None,
          padding: int | Sequence[int], dims: int) -> Any:
    """Pooling over the last dims (1 or 2) dimensions by the max or avg pooling stencil"""
    name = f"{op}_pool{dims}d"
    xa = _float_array(x, 'x')
    if xa.ndim not in (dims + 1, dims + 2):
        raise ValueError(f"{name} requires a {dims + 1}D or {dims + 2}D input, got shape {xa.shape}")
    kernel = _pair(kernel_size, 'kernel_size', 1)
    step = kernel if stride is None else _pair(stride, 'stride', 1)
    pad = _pair(padding, 'padding', 0)
    if dims == 1:
        kernel, step, pad = (1, kernel[1]), (1, step[1]), (0, pad[1])
    if any(2 * p > k for p, k in zip(pad, kernel)):
        raise ValueError(f"padding {padding} must be at most half the kernel size {kernel_size}")

    h, w = (1, xa.shape[-1]) if dims == 1 else xa.shape[-2:]
    oh, ow = ((size + 2 * p - k) // s + 1 for size, k, s, p in zip((h, w), kernel, step, pad))
    if oh < 1 or ow < 1:
        raise ValueError(f"Kernel of size {kernel_size} is larger than the padded input of shape {xa.shape[-dims:]}")

    lead = xa.shape[:-dims]
    params = array([_size(lead), h, w, *kernel, oh, ow, *step, *pad], 'int')
    node = _add_array_op(f"{op}pool2d_floatarr", [xa.net, params.net], _size(lead) * oh * ow)
    assert isinstance(node.result, ArrayNet)
    ret = array._from_net(node.result, (*lead, *((oh, ow)[2 - dims:])))
    return tensor._from_array(ret) if isinstance(x, tensor) else ret


def max_pool2d(x: TArr, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None,
               padding: int | Sequence[int] = 0) -> TArr:
    """2D max pooling: the largest element of each window, computed by a
    single array stencil.

    Arguments:
        x: Input of shape (channels, height, width) or (batch, channels, height, width).
        kernel_size: Size of the window, a single integer or (vertical, horizontal).
        stride: Step of the window, a single integer or (vertical, horizontal).
            The default is the kernel size (windows without overlap).
        padding: Padding on both sides of the input, a single integer or
            (vertical, horizontal), at most half the kernel size. Padded
            elements are ignored.

    Returns:
        Float array or tensor (same type as x) of shape (channels, out_height, out_width)
        with a leading batch dimension if x has one.
    """
    ret: TArr = _pool(x, 'max', kernel_size, stride, padding, 2)
    return ret


def avg_pool2d(x: TArr, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None,
               padding: int | Sequence[int] = 0) -> TArr:
    """2D average pooling: the mean of each window, computed by a single
    array stencil.

    Arguments:
        x: Input of shape (channels, height, width) or (batch, channels, height, width).
        kernel_size: Size of the window, a single integer or (vertical, horizontal).
        stride: Step of the window, a single integer or (vertical, horizontal).
            The default is the kernel size (windows without overlap).
        padding: Zero padding on both sides of the input, a single integer or
            (vertical, horizontal), at most half the kernel size. Padded
            elements count as zeros.

    Returns:
        Float array or tensor (same type as x) of shape (channels, out_height, out_width)
        with a leading batch dimension if x has one.
    """
    ret: TArr = _pool(x, 'avg', kernel_size, stride, padding, 2)
    return ret


def max_pool1d(x: TArr, kernel_size: int, stride: int | None = None, padding: int = 0) -> TArr:
    """1D max pooling: the largest element of each window, computed by a
    single array stencil.

    Arguments:
        x: Input of shape (channels, length) or (batch, channels, length).
        kernel_size: Size of the window.
        stride: Step of the window, the default is the kernel size.
        padding: Padding on both sides of the input, at most half the kernel
            size. Padded elements are ignored.

    Returns:
        Float array or tensor (same type as x) of shape (channels, out_length)
        with a leading batch dimension if x has one.
    """
    ret: TArr = _pool(x, 'max', kernel_size, stride, padding, 1)
    return ret


def avg_pool1d(x: TArr, kernel_size: int, stride: int | None = None, padding: int = 0) -> TArr:
    """1D average pooling: the mean of each window, computed by a single
    array stencil.

    Arguments:
        x: Input of shape (channels, length) or (batch, channels, length).
        kernel_size: Size of the window.
        stride: Step of the window, the default is the kernel size.
        padding: Zero padding on both sides of the input, at most half the
            kernel size. Padded elements count as zeros.

    Returns:
        Float array or tensor (same type as x) of shape (channels, out_length)
        with a leading batch dimension if x has one.
    """
    ret: TArr = _pool(x, 'avg', kernel_size, stride, padding, 1)
    return ret


def global_avg_pool2d(x: TArr) -> TArr:
    """Mean of each channel over height and width.

    Arguments:
        x: Input of shape (channels, height, width) or (batch, channels, height, width).

    Returns:
        Float array or tensor (same type as x) of shape (channels,)
        with a leading batch dimension if x has one.
    """
    pooled: Any = _pool(x, 'avg', x.shape[-2:], None, 0, 2)
    ret: TArr = pooled.reshape(*pooled.shape[:-2])
    return ret
