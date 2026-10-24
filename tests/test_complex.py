import cmath
import itertools
import operator
from typing import Any, Callable, Generator

import pytest

import copapy as cp
from copapy import Target, cxvalue, cxtensor
from runner_helpers import trig_tol


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def approx(z: Any, abs: float = 1e-5) -> Any:
    return pytest.approx(z, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def variable(z: complex) -> cxvalue:
    return cxvalue(cp.value(z.real), cp.value(z.imag))


def to_complex(t: cxtensor) -> list[complex]:
    """Flat elements of a cxtensor of constants"""
    return [complex(r, i) for r, i in zip(t.real.values, t.imag.values)]


def variable_tensor(rows: list[list[complex]]) -> cxtensor:
    return cxtensor(cp.tensor([[cp.value(z.real) for z in r] for r in rows]),
                    cp.tensor([[cp.value(z.imag) for z in r] for r in rows]))


def matmul_ref(a: list[list[complex]], b: list[list[complex]]) -> list[complex]:
    return [sum(a[i][k] * b[k][j] for k in range(len(b))) for i in range(len(a)) for j in range(len(b[0]))]


@pytest.fixture(params=[False, True], ids=['scalar', 'packed'])
def packed(request: pytest.FixtureRequest) -> Generator[bool, None, None]:
    threshold = cp.tensor.pack_threshold
    if request.param:
        cp.tensor.pack_threshold = 1
    yield request.param
    cp.tensor.pack_threshold = threshold


Z1 = 1.5 - 2.0j
Z2 = -0.5 + 3.0j
A = [[1 + 2j, -1j], [0.5, 2 - 1j]]
B = [[2 - 1j, 1 + 1j], [-1j, 3]]

BINARY_OPS: list[Callable[[Any, Any], Any]] = [operator.add, operator.sub, operator.mul, operator.truediv]


def test_constructor():
    assert (cxvalue(1, 2).real, cxvalue(1, 2).imag) == (1.0, 2.0)
    assert (cxvalue(3 - 4j).real, cxvalue(3 - 4j).imag) == (3.0, -4.0)
    assert isinstance(cxvalue(1).real, float)


@pytest.mark.parametrize("op", BINARY_OPS)
def test_scalar_ops(op: Callable[[Any, Any], Any]):
    a, b = variable(Z1), variable(Z2)
    exprs = [op(a, b), op(a, Z2), op(Z1, b), op(a, 2.0), op(2.0, b), op(a, cp.value(2.0))]
    refs = [op(Z1, Z2), op(Z1, Z2), op(Z1, Z2), op(Z1, 2.0), op(2.0, Z2), op(Z1, 2.0)]
    for result, ref in zip(evaluate(*exprs), refs):
        assert result == approx(ref)


def test_constant_scalar_ops_are_folded():
    assert isinstance((cxvalue(Z1) * Z2).real, float)
    assert cxvalue(1, 2) * 1j == cxvalue(-2, 1)


def test_real_operand_generates_no_imaginary_code():
    x = cp.value(2.0)
    z = cxvalue(x) * 3.0
    assert isinstance(z.imag, float) and z.imag == 0.0
    z = cxvalue(x) * cxvalue(cp.value(1.0))
    assert z.imag == 0.0


def test_scalar_functions():
    a = variable(Z1)
    tol = trig_tol()
    ab, ang, ex, lg, conj, neg, p3, pm2 = evaluate(abs(a), a.angle(), a.exp(), a.log(), a.conj(), -a, a ** 3, a ** -2)
    assert ab == approx(abs(Z1))
    assert ang == approx(cmath.phase(Z1), tol)
    assert ex == approx(cmath.exp(Z1), tol * 10)
    assert lg == approx(cmath.log(Z1), tol)
    assert conj == approx(Z1.conjugate())
    assert neg == approx(-Z1)
    assert p3 == approx(Z1 ** 3, 1e-4)
    assert pm2 == approx(Z1 ** -2)


def test_from_polar():
    z, = evaluate(cxvalue.from_polar(cp.value(2.0), cp.value(0.5)))
    assert z == approx(cmath.rect(2.0, 0.5), trig_tol())


def test_scalar_compare():
    a = variable(Z1)
    eq, ne, eq2 = evaluate(a == Z1, a != Z1, a == Z2)
    assert (eq, ne, eq2) == (1, 0, 0)


def test_grad():
    x, y = cp.value(1.5), cp.value(-2.0)
    loss = (cxvalue(x, y) * cxvalue(Z2)).abs2()
    dx, dy = evaluate(cp.grad(loss, x), cp.grad(loss, y))
    # |z * w|^2 = |w|^2 * (x^2 + y^2)
    assert dx == pytest.approx(2 * abs(Z2) ** 2 * 1.5, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
    assert dy == pytest.approx(2 * abs(Z2) ** 2 * -2.0, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_tensor_constructor():
    t = cxtensor(A)
    assert t.shape == (2, 2)
    assert to_complex(t) == [z for row in A for z in row]
    assert cxtensor([[1, 2], [3, 4]])._im is None
    assert cxtensor([cxvalue(1, 2), 3j]).shape == (2,)
    with pytest.raises(ValueError):
        cxtensor(cp.tensor([1.0, 2.0]), cp.tensor([1.0]))


@pytest.mark.parametrize("op", BINARY_OPS)
def test_tensor_elementwise(op: Callable[[Any, Any], Any], packed: bool):
    a, b = variable_tensor(A), variable_tensor(B)
    exprs = [op(a, b), op(a, Z2), op(Z1, b), op(a, cp.tensor([[1.0, 2.0], [3.0, 4.0]])), op(a, cxvalue(cp.value(2.0), cp.value(1.0)))]
    reals = [[1.0, 2.0], [3.0, 4.0]]
    refs = [[op(x, y) for ra, rb in zip(A, B) for x, y in zip(ra, rb)],
            [op(x, Z2) for r in A for x in r],
            [op(Z1, y) for r in B for y in r],
            [op(x, y) for ra, rb in zip(A, reals) for x, y in zip(ra, rb)],
            [op(x, 2 + 1j) for r in A for x in r]]
    for result, ref in zip(evaluate(*exprs), refs):
        assert to_complex(result) == approx(ref)


def test_tensor_reflected_ops(packed: bool):
    a = variable_tensor(A)
    reals = [[1.0, 2.0], [3.0, 4.0]]
    t = cp.tensor([[cp.value(v) for v in r] for r in reals])
    s, p, m = evaluate(t + a, t * a, t @ a)
    assert to_complex(s) == approx([x + y for ra, rb in zip(reals, A) for x, y in zip(ra, rb)])
    assert to_complex(p) == approx([x * y for ra, rb in zip(reals, A) for x, y in zip(ra, rb)])
    assert to_complex(m) == approx(matmul_ref([[complex(v) for v in r] for r in reals], A))


def test_matmul(packed: bool):
    a, b = variable_tensor(A), variable_tensor(B)
    ab, ah, av = evaluate(a @ b, a.H @ b, a @ cp.vector([cp.value(1.0), cp.value(2.0)]))
    assert to_complex(ab) == approx(matmul_ref(A, B))
    AH = [[A[j][i].conjugate() for j in range(2)] for i in range(2)]
    assert to_complex(ah) == approx(matmul_ref(AH, B))
    assert to_complex(av) == approx([A[i][0] + 2 * A[i][1] for i in range(2)])


def test_real_tensor_matmul_needs_two_products():
    """Real tensor times complex tensor: the imaginary part of the real tensor generates no code"""
    a = cxtensor(cp.tensor([[cp.value(1.0), cp.value(2.0)], [cp.value(3.0), cp.value(4.0)]]))
    assert a._im is None
    assert (a @ a)._im is None
    assert (a @ variable_tensor(B))._im is not None


def test_tensor_shape_ops_and_reductions(packed: bool):
    a = variable_tensor(A)
    total, row_sums, elem, tr, flat, absval, ang = evaluate(
        a.sum(), a.sum(axis=1), a[1, 0], a.T, a.flatten(), abs(a), a.angle())
    assert total == approx(sum(z for r in A for z in r))
    assert to_complex(row_sums) == approx([sum(r) for r in A])
    assert to_complex(elem) == approx([A[1][0]])
    assert to_complex(tr) == approx([A[j][i] for i in range(2) for j in range(2)])
    assert to_complex(flat) == approx([z for r in A for z in r])
    assert list(absval.values) == approx([abs(z) for r in A for z in r])
    assert list(ang.values) == approx([cmath.phase(z) for r in A for z in r], trig_tol())
    assert variable_tensor(A).get_scalar(0, 1).imag is not None


def test_tensor_exp(packed: bool):
    a = variable_tensor(A)
    ex, = evaluate(a.exp())
    assert to_complex(ex) == approx([cmath.exp(z) for r in A for z in r], trig_tol() * 20)


# Real vectors and tensors combined with complex scalars and tensors

R1 = [1.5, -2.0]
R2 = [[1.0, 2.0], [-0.5, 3.0]]
C1 = [1 + 2j, -1j]

REAL_OPERANDS = ['vector', 'tensor']
COMPLEX_OPERANDS = ['cxvalue', 'cxtensor1d', 'cxtensor2d']

# Pure Python data of the operands
DATA: dict[str, Any] = {'vector': R1, 'tensor': R2, 'cxvalue': Z1, 'cxtensor1d': C1, 'cxtensor2d': A}


def operand(kind: str) -> Any:
    """Copapy operand with variable elements"""
    if kind == 'vector':
        return cp.vector([cp.value(x) for x in R1])
    if kind == 'tensor':
        return cp.tensor([[cp.value(x) for x in r] for r in R2])
    if kind == 'cxvalue':
        return variable(Z1)
    if kind == 'cxtensor1d':
        return cxtensor(cp.tensor([cp.value(z.real) for z in C1]), cp.tensor([cp.value(z.imag) for z in C1]))
    return variable_tensor(A)


def shape_of(x: Any) -> tuple[int, ...]:
    return (len(x),) + shape_of(x[0]) if isinstance(x, list) else ()


def element(x: Any, shape: tuple[int, ...], index: tuple[int, ...]) -> Any:
    """Element of x for the index into the broadcast shape (numpy rules)"""
    for dim, i in zip(shape, index[len(index) - len(shape):]):
        x = x[0 if dim == 1 else i]
    return x


def elementwise_ref(op: Callable[[Any, Any], Any], a: Any, b: Any) -> list[complex]:
    sa, sb = shape_of(a), shape_of(b)
    ndim = max(len(sa), len(sb))
    pa, pb = (1,) * (ndim - len(sa)) + sa, (1,) * (ndim - len(sb)) + sb
    shape = tuple(max(da, db) for da, db in zip(pa, pb))
    return [op(element(a, sa, i), element(b, sb, i)) for i in itertools.product(*(range(d) for d in shape))]


def matmul_any_ref(a: Any, b: Any) -> list[complex]:
    """a @ b for 1D and 2D operands, flat"""
    a2 = a if isinstance(a[0], list) else [a]
    b2 = b if isinstance(b[0], list) else [[x] for x in b]
    return matmul_ref(a2, b2)


def complex_mix_ids(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


MIXED_PAIRS = [(r, c) for r in REAL_OPERANDS for c in COMPLEX_OPERANDS] +               [(c, r) for r in REAL_OPERANDS for c in COMPLEX_OPERANDS]


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("pair", MIXED_PAIRS, ids=complex_mix_ids)
def test_real_complex_elementwise(op: Callable[[Any, Any], Any], pair: tuple[str, str], packed: bool):
    a, b = operand(pair[0]), operand(pair[1])
    ref = elementwise_ref(op, DATA[pair[0]], DATA[pair[1]])
    result = op(a, b)
    assert isinstance(result, cxtensor)
    value, = evaluate(result)
    assert to_complex(value) == approx(ref)


MATMUL_PAIRS = [(r, c) for r in REAL_OPERANDS for c in ('cxtensor1d', 'cxtensor2d')] +                [(c, r) for r in REAL_OPERANDS for c in ('cxtensor1d', 'cxtensor2d')] +                [('cxtensor1d', 'cxtensor2d'), ('cxtensor2d', 'cxtensor1d'), ('cxtensor2d', 'cxtensor2d')]


@pytest.mark.parametrize("matmul", [operator.matmul, cp.matmul], ids=['operator', 'function'])
@pytest.mark.parametrize("pair", MATMUL_PAIRS, ids=complex_mix_ids)
def test_real_complex_matmul(matmul: Callable[[Any, Any], Any], pair: tuple[str, str], packed: bool):
    a, b = operand(pair[0]), operand(pair[1])
    result = matmul(a, b)
    assert isinstance(result, cxtensor)
    value, = evaluate(result)
    assert to_complex(value) == approx(matmul_any_ref(DATA[pair[0]], DATA[pair[1]]))


@pytest.mark.parametrize("matmul", [operator.matmul, cp.matmul], ids=['operator', 'function'])
@pytest.mark.parametrize("pair", [('vector', 'cxvalue'), ('tensor', 'cxvalue'), ('cxvalue', 'vector'),
                                  ('cxvalue', 'tensor'), ('cxtensor2d', 'cxvalue'), ('cxvalue', 'cxtensor2d'),
                                  ('tensor', 'float'), ('cxtensor2d', 'float')], ids=complex_mix_ids)
def test_matmul_with_scalar_raises(matmul: Callable[[Any, Any], Any], pair: tuple[str, str]):
    a, b = (2.0 if k == 'float' else operand(k) for k in pair)
    with pytest.raises(TypeError):
        matmul(a, b)


def test_real_matmul_function(packed: bool):
    t = cp.tensor([[cp.value(x) for x in r] for r in R2])
    v = cp.vector([cp.value(x) for x in R1])
    exprs = cp.matmul(t, t), cp.matmul(t, v), cp.matmul(v, t), cp.matmul(v, v)
    assert [type(e) for e in exprs[:3]] == [cp.tensor, cp.vector, cp.vector]
    assert isinstance(exprs[3], cp.value)
    tt, tv, vt, vv = evaluate(*exprs)
    assert list(tt.values) == approx(matmul_any_ref(R2, R2))
    assert list(tv) == approx(matmul_any_ref(R2, R1))
    assert list(vt) == approx(matmul_any_ref(R1, R2))
    assert vv == approx(sum(x * x for x in R1))


def test_matmul_function_arrays():
    a2, a1 = cp.array(R2), cp.array(R1)
    exprs = cp.matmul(a2, a2), cp.matmul(a2, a1), cp.matmul(a1, a2), cp.matmul(a1, a1)
    assert [type(e) for e in exprs] == [cp.array, cp.array, cp.array, cp.value]
    assert [e.shape for e in exprs[:3]] == [(2, 2), (2,), (2,)]
    aa, av, va, vv = evaluate(*exprs)
    assert [x for r in aa for x in r] == approx(matmul_any_ref(R2, R2))
    assert av == approx(matmul_any_ref(R2, R1))
    assert va == approx(matmul_any_ref(R1, R2))
    assert vv == approx(sum(x * x for x in R1))


@pytest.mark.parametrize("pair", [
    (cp.tensor(R2), cp.vector([1.0, 2.0, 3.0])),
    (cp.vector([1.0, 2.0, 3.0]), cp.tensor(R2)),
    (cxtensor(A), cp.vector([1.0, 2.0, 3.0])),
    (cp.vector([1.0, 2.0, 3.0]), cxtensor(A)),
    (cp.array(R2), cp.array([1.0, 2.0, 3.0])),
    (cp.array([1.0, 2.0, 3.0]), cp.array(R2)),
], ids=['tensor-vector', 'vector-tensor', 'cxtensor-vector', 'vector-cxtensor', 'array2d-array1d', 'array1d-array2d'])
def test_matmul_function_shape_mismatch(pair: tuple[Any, Any]):
    with pytest.raises(ValueError):
        cp.matmul(*pair)


def test_tensor_has_no_matmul_method():
    assert not hasattr(cp.tensor, 'matmul')
    assert not hasattr(cxtensor, 'matmul')
    assert not hasattr(cp.array, 'matmul')
