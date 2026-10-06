"""Indexing of arrays, vectors and tensors by copapy values (computed at runtime)"""
from typing import Any

import pytest

import copapy as cp
import copapy._compiler as compiler
from copapy.backend import Store


def clamp(k: int, n: int) -> int:
    """Reference: negative indices count from the end, then clamped to [0, n - 1]"""
    return min(max(k + n if k < 0 else k, 0), n - 1)


DATA = [[float(r * 10 + c) for c in range(4)] for r in range(3)]
INDICES = [(0, 0), (1, 2), (2, 3), (-1, -1), (-2, 1), (5, 9), (-7, -9)]


@pytest.mark.parametrize('kind', ['array', 'tensor'])
def test_runtime_indices(kind: str) -> None:
    i, j = cp.value(0), cp.value(0)
    x: Any = cp.array(DATA) if kind == 'array' else cp.tensor([[cp.value(v) for v in row] for row in DATA])
    exprs = [x[i, j], x[i], x[:, j], x[i, 1:3], x[1, j]]

    tg = cp.Target()
    tg.compile(*exprs)
    for vi, vj in INDICES:
        tg.write_value(i, vi)
        tg.write_value(j, vj)
        tg.run()
        r, c = clamp(vi, 3), clamp(vj, 4)
        out = [tg.read_value(e) for e in exprs]
        if kind == 'tensor':
            out = [list(o.values) for o in out]  # read 0-d tensors have shape (1,)
        expected = [DATA[r][c], DATA[r], [row[c] for row in DATA], DATA[r][1:3], DATA[1][c]]
        for o, e in zip(out, expected):
            assert (o if isinstance(o, list) else [o]) == (e if isinstance(e, list) else [e]), (vi, vj)


def test_vector_lookup_table() -> None:
    """A constant vector is a lookup table: no code for the table itself"""
    table = cp.vector([0.0, 1.5, 4.0, 9.5, 16.0, 25.5])
    k = cp.value(0)
    y = table[k] * 2.0

    ordered = compiler.stable_toposort(compiler.get_all_dag_edges([Store(y.net)]))
    assert [n.name for n in ordered].count('copy_arr') == 1
    assert any(isinstance(n, compiler.ArrayConst) for n in ordered)  # the table is data
    pack_stores = [node for _, node in compiler.add_load_ops(ordered) if node.name == compiler.PACK_STORE]
    assert len(pack_stores) == 1  # only the offset of the copy is stored at runtime

    tg = cp.Target()
    tg.compile(y)
    for vk in range(-8, 9):
        tg.write_value(k, vk)
        tg.run()
        assert tg.read_value(y) == table.values[clamp(vk, 6)] * 2.0


def test_index_types() -> None:
    v = cp.vector(cp.value(float(x)) for x in [3, 4, 5])
    flag = cp.value(1.0) > 0.0  # bool value: index 1
    res = v[flag]
    tg = cp.Target()
    tg.compile(res)
    tg.run()
    out = tg.read_value(res)
    assert out == 4.0

    with pytest.raises(TypeError):
        v[cp.value(1.0)]  # float index


def test_pack_constants_at_load_time() -> None:
    """Constant elements of a packed array are written when loading the program,
    only computed elements and copapy values are stored on each run"""
    x = cp.value(2.0)
    p = cp.array([1.5, x, 3.0, 4.0, x * 2.0, 6.0, 7.0, 8.0])
    ordered = compiler.stable_toposort(compiler.get_all_dag_edges([p.net.source]))
    pack_stores = [node for _, node in compiler.add_load_ops(ordered) if node.name == compiler.PACK_STORE]
    assert len(pack_stores) == 2  # x and x * 2.0

    tg = cp.Target()
    tg.compile(p)
    tg.run()
    assert tg.read_value(p) == [1.5, 2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0]
    tg.write_value(x, 10.0)
    tg.run()
    assert tg.read_value(p) == [1.5, 10.0, 3.0, 4.0, 20.0, 6.0, 7.0, 8.0]

    # Only constants: no stores at all, the array is still allocated and initialized
    c = cp.array([cp.value(1.0) * 0.0 + 1.0, 2.0])[cp.value(1)]
    tg2 = cp.Target()
    tg2.compile(c)
    tg2.run()
    assert tg2.read_value(c) == 2.0
