import copapy as cp
import pytest


def test_vectors_init():
    tt1 = cp.vector(range(3)) + cp.vector([1.1, 2.2, 3.3])
    tt2 = cp.vector([1.1, 2, cp.variable(5)]) + cp.vector(range(3))
    tt3 = (cp.vector(range(3)) + 5.6)
    tt4 = cp.vector([1.1, 2, 3]) + cp.vector(cp.variable(v) for v in range(3))
    tt5 = cp.vector([1, 2, 3]).dot(tt4)

    print(tt1, tt2, tt3, tt4, tt5)


def test_compiled_vectors():
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.variable(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.variable(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    tg = cp.Target()
    tg.compile(t2, t4, t5)
    tg.run()

    assert isinstance(t2, cp.variable)
    assert tg.read_value(t2) == 10 + 11 + 12 + 0 + 1 + 2

    assert isinstance(t4, cp.variable)
    assert tg.read_value(t4) == pytest.approx(((10/1*2) + (12/2*2) + (14/3*2)), 0.001)  # pyright: ignore[reportUnknownMemberType]

    assert isinstance(t5, cp.variable)
    assert tg.read_value(t5) == pytest.approx(((10/1*2)**2 + (12/2*2)**2 + (14/3*2)**2) ** 0.5, 0.001)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_compiled_vectors()
