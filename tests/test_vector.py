import math
import copapy as cp
import pytest
from copapy import filters

def test_vectors_init():
    tt1 = cp.vector(range(3)) + cp.vector([1.1, 2.2, 3.3])
    tt2 = cp.vector([1.1, 2, cp.value(5)]) + cp.vector(range(3))
    tt3 = (cp.vector(range(3)) + 5.6)
    tt4 = cp.vector([1.1, 2, 3]) + cp.vector(cp.value(v) for v in range(3))
    tt5 = cp.vector([1, 2, 3]).dot(tt4)

    print(tt1, tt2, tt3, tt4, tt5)


def test_compiled_vectors():
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.value(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.value(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    t6 = cp.angle_between(cp.vector([cp.value(5.0), 0.0, 0.0]), cp.vector([5.0, 5.0, 0.0]))

    tg = cp.Target()
    tg.compile(t2, t4, t5, t6)
    tg.run()

    assert isinstance(t2, cp.value)
    assert tg.read_value(t2) == 10 + 11 + 12 + 0 + 1 + 2

    assert isinstance(t4, cp.value)
    assert tg.read_value(t4) == pytest.approx(((10/1*2) + (12/2*2) + (14/3*2)), 0.001)  # pyright: ignore[reportUnknownMemberType]

    assert isinstance(t5, cp.value)
    assert tg.read_value(t5) == pytest.approx(((10/1*2)**2 + (12/2*2)**2 + (14/3*2)**2) ** 0.5, 0.001)  # pyright: ignore[reportUnknownMemberType]

    assert isinstance(t6, cp.value)
    assert tg.read_value(t6) == pytest.approx(math.pi / 4, 0.001), tg.read_value(t6)  # pyright: ignore[reportUnknownMemberType]


def test_non_compiled_vector_operations():
    v1 = cp.vector([1.0, 2.0, 3.0])
    v2 = cp.vector([4.0, 5.0, 6.0])

    # dot product
    assert v1.dot(v2) == pytest.approx(1*4 + 2*5 + 3*6)  # pyright: ignore[reportUnknownMemberType]

    # cross product
    cross = v1.cross(v2)
    assert isinstance(cross, cp.vector)
    assert cross.values == pytest.approx((-3.0, 6.0, -3.0))  # pyright: ignore[reportUnknownMemberType]

    # magnitude
    mag = v1.magnitude()
    assert mag == pytest.approx((1**2 + 2**2 + 3**2) ** 0.5)  # pyright: ignore[reportUnknownMemberType]

    # normalize
    norm = v1.normalize()
    norm_mag = norm.magnitude()
    assert norm_mag == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    # distance
    dist = cp.distance(v1, v2)
    assert dist == pytest.approx(((1-4)**2 + (2-5)**2 + (3-6)**2) ** 0.5)  # pyright: ignore[reportUnknownMemberType]

    # scalar projection
    scalar_proj = cp.scalar_projection(v1, v2)
    expected_scalar_proj = v1 @ v2 / (v2.magnitude() + 1e-20)
    assert scalar_proj == pytest.approx(expected_scalar_proj)  # pyright: ignore[reportUnknownMemberType]

    # vector projection
    vector_proj = cp.vector_projection(v1, v2)
    assert isinstance(vector_proj, cp.vector)
    # Check direction and magnitude
    proj_mag = vector_proj.magnitude()
    tmp = cp.abs(scalar_proj)
    assert proj_mag == pytest.approx(tmp)  # pyright: ignore[reportUnknownMemberType]

    # angle between
    angle = cp.angle_between(v1, v2)
    import math
    expected_angle = cp.acos(v1 @ v2 / ((v1.magnitude()) * (v2.magnitude()) + 1e-20))
    assert angle == pytest.approx(expected_angle)  # pyright: ignore[reportUnknownMemberType]

    # rotate_vector
    axis = cp.vector([0.0, 0.0, 1.0])
    angle_rad = math.pi / 2
    rotated = cp.rotate_vector(v1, axis, angle_rad)
    assert isinstance(rotated, cp.vector)
    # Rotating [1,2,3] 90deg around z should give [-2,1,3] (approx)
    assert rotated.values[0] == pytest.approx(-2.0, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]
    assert rotated.values[1] == pytest.approx(1.0, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]
    assert rotated.values[2] == pytest.approx(3.0, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_sort_vector():
    vlist = [50, 21, 20, 10, 22, 1, 80, 70, 90]
    t1 = cp.vector(cp.value(v) for v in vlist)
    #t1 = cp.vector(v for v in vlist)

    t2 = filters.median(t1)

    tg = cp.Target()
    tg.compile(t2)
    tg.run()

    result = tg.read_value(t2)

    ref = sorted(vlist)[len(vlist) // 2]
    print(sorted(vlist))

    assert ref == result


if __name__ == "__main__":
    #test_vectors_init()
    #test_compiled_vectors()
    test_sort_vector()
    print('Finished!')
