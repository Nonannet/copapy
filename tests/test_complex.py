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


# Mixing with Python complex numbers, math functions, comparison and delay

def test_value_with_python_complex():
    x, y = cp.value(1.5), cp.value(-2.0)
    exprs = [x + 2j, 2j + x, x * 1j, x - 1j, 1j - x, x / (1 + 1j), (1 + 1j) / x, x + 1j * y, x == 1.5 + 0j, x != 1.5 + 0j]
    refs = [1.5 + 2j, 1.5 + 2j, 1.5j, 1.5 - 1j, 1j - 1.5, 1.5 / (1 + 1j), (1 + 1j) / 1.5, Z1, 1, 0]
    assert all(isinstance(e, cxvalue) for e in exprs[:8])
    for result, ref in zip(evaluate(*exprs), refs):
        assert result == approx(ref)
    with pytest.raises(TypeError):
        x > 1j  # type: ignore[operator]
    with pytest.raises(TypeError):
        x & 1j  # type: ignore[operator]


def test_constructor_rejects_imag_for_complex():
    with pytest.raises(TypeError):
        cxvalue(1j, 5.0)
    assert cxvalue(1j, 0.0) == cxvalue(0, 1)


def test_math_functions_dispatch():
    a = variable(Z1)
    tol = trig_tol() * 10
    results = evaluate(cp.exp(a), cp.log(a), cp.sqrt(a), cp.abs(a), cp.pow(a, 3), cp.pow(a, 0.5),
                       cp.pow(2.0, a), cp.pow(a, Z2), a ** cp.value(1.5), 2 ** a, Z2 ** a,
                       cp.sqrt(cxvalue(cp.value(-4.0))), cp.sqrt(cxvalue(cp.value(0.0), cp.value(0.0))))
    refs = [cmath.exp(Z1), cmath.log(Z1), cmath.sqrt(Z1), abs(Z1), Z1 ** 3, Z1 ** 0.5,
            2.0 ** Z1, Z1 ** Z2, Z1 ** 1.5, 2 ** Z1, Z2 ** Z1, 2j, 0j]
    for result, ref in zip(results, refs):
        assert result == approx(ref, max(tol, 1e-4) * max(1.0, abs(ref)))


def test_tensor_math_functions(packed: bool):
    a = variable_tensor(A)
    flat = [z for r in A for z in r]
    tol = max(trig_tol() * 20, 1e-4)
    lg, sq, p3, pm1, ph, pc, rp, mean, row_mean, ex, ab = evaluate(
        cp.log(a), cp.sqrt(a), a ** 3, a ** -1, a ** 0.5, a ** Z2, 2.0 ** a, a.mean(), a.mean(axis=1), cp.exp(a), cp.abs(a))
    assert to_complex(lg) == approx([cmath.log(z) for z in flat], tol)
    assert to_complex(sq) == approx([cmath.sqrt(z) for z in flat], tol)
    assert to_complex(p3) == approx([z ** 3 for z in flat], tol * 10)
    assert to_complex(pm1) == approx([1 / z for z in flat], tol)
    assert to_complex(ph) == approx([z ** 0.5 for z in flat], tol)
    assert to_complex(pc) == approx([z ** Z2 for z in flat], tol * 100)
    assert to_complex(rp) == approx([2.0 ** z for z in flat], tol * 10)
    assert mean == approx(sum(flat) / 4)
    assert to_complex(row_mean) == approx([sum(r) / 2 for r in A])
    assert to_complex(ex) == approx([cmath.exp(z) for z in flat], tol)
    assert list(ab.values) == approx([abs(z) for z in flat])
    assert to_complex(cxtensor(A) ** 0) == [1.0] * 4


def test_tensor_compare(packed: bool):
    a = variable_tensor(A)
    other = [[1 + 2j, 1j], [0.5, 2 - 1j]]
    eq, ne, eq_scalar, eq_real, eq_value = evaluate(
        a == cxtensor(other), a != cxtensor(other), a == 0.5, cxtensor([[cp.value(1.0), 2.0]]) == cp.tensor([[1.0, 3.0]]),
        variable(0.5 + 0j) == a)
    assert list(eq.values) == [1, 0, 1, 1]
    assert list(ne.values) == [0, 1, 0, 0]
    assert list(eq_scalar.values) == [0, 0, 1, 0]
    assert list(eq_real.values) == [1, 0]
    assert list(eq_value.values) == [0, 0, 1, 0]


def test_tensor_from_array_and_indexing():
    t = cxtensor(cp.array([1.0, 2.0]), cp.array([3.0, 4.0]))
    real = cxtensor(cp.array([1.0, 2.0]))
    assert real._im is None
    read, read_real = evaluate(t, real)
    assert to_complex(read) == [1 + 3j, 2 + 4j]
    assert to_complex(read_real) == [1 + 0j, 2 + 0j]
    # Like a tensor: single elements are 0-d, item() gives the scalar
    assert t[0].shape == () and [e.shape for e in t] == [(), ()]
    assert isinstance(t[0].item(), cxvalue)


def test_delay():
    x = cp.value(0.0)
    z = cxvalue(x, x * 2.0)
    t = cxtensor([cxvalue(x, x), cxvalue(x * 3.0)])
    real = cxtensor([x, x * 2.0])
    outputs = [cp.delay(z, init=1 - 1j), cp.delay(t), cp.delay(real, n=2), cp.delay(real, init=1j)]
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task(outputs, period=0.01))
    results: list[Any] = []
    for k, v in enumerate([1.0, 2.0, 3.0]):
        tg.write_value(x, v)
        tg.step((k + 1) * 0.01)
        results.append(tg.read_value(outputs))
    assert [r[0] for r in results] == [1 - 1j, 1 + 2j, 2 + 4j]
    assert [to_complex(r[1]) for r in results] == [[0j, 0j], [1 + 1j, 3 + 0j], [2 + 2j, 6 + 0j]]
    assert [to_complex(r[2]) for r in results] == [[0j, 0j], [0j, 0j], [1 + 0j, 2 + 0j]]
    assert [to_complex(r[3]) for r in results] == [[1j, 1j], [1 + 0j, 2 + 0j], [2 + 0j, 4 + 0j]]
    assert real._im is None and cp.delay(real)._im is None


def test_delay_structures_and_inits():
    x = cp.value(0.0)
    z = cxvalue(x)
    t = cxtensor([cxvalue(x, x), cxvalue(x * 3.0)])
    # A constant part with the same initial value stays a constant
    assert cp.delay(z).imag == 0.0 and isinstance(cp.delay(z, n=3).imag, float)
    assert isinstance(cp.delay(z, init=1j).imag, cp.value)
    outputs = [cp.delay([z, x], init=[2j, 5.0]), cp.delay((z, t)), cp.delay(t, init=[1 + 1j, 2]),
               cp.delay(t, init=cp.tensor([4.0, 5.0]))]
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task(outputs, period=0.01))
    results: list[Any] = []
    for k, v in enumerate([1.0, 2.0]):
        tg.write_value(x, v)
        tg.step((k + 1) * 0.01)
        results.append(tg.read_value(outputs))
    assert [r[0] for r in results] == [[2j, 5.0], [1 + 0j, 1.0]]
    assert [(r[1][0], to_complex(r[1][1])) for r in results] == [(0j, [0j, 0j]), (1 + 0j, [1 + 1j, 3 + 0j])]
    assert [to_complex(r[2]) for r in results] == [[1 + 1j, 2 + 0j], [1 + 1j, 3 + 0j]]
    assert [to_complex(r[3]) for r in results] == [[4 + 0j, 5 + 0j], [1 + 1j, 3 + 0j]]
    with pytest.raises(TypeError):
        cp.delay(z, init=x)
    with pytest.raises(ValueError):
        cp.delay(t, init=[1j, 2j, 3j])
