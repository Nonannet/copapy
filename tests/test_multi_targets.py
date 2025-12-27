import copapy as cp
import pytest

def test_multi_target():
    # Define variables
    a = cp.value(0.25)
    b = cp.value(0.87)

    # Define computations
    c = a + b * 2.0
    d = c ** 2 + cp.sin(a)
    e = d + cp.sqrt(b)

    # Create a target, compile and run
    tg1 = cp.Target()
    tg1.compile(e)

    # Patch constant value
    a.net.source = cp._basic_types.CPConstant(1000.0)

    tg2 = cp.Target()
    tg2.compile(e)

    tg1.run()
    tg2.run()

    print("Result tg1:", tg1.read_value(e))
    print("Result tg2:", tg2.read_value(e))

    # Assertions to verify correctness
    assert tg1.read_value(e) == pytest.approx((0.25 + 0.87 * 2.0) ** 2 + cp.sin(0.25) + cp.sqrt(0.87), 0.005)  # pyright: ignore[reportUnknownMemberType]
    assert tg2.read_value(e) == pytest.approx((1000.0 + 0.87 * 2.0) ** 2 + cp.sin(1000.0) + cp.sqrt(0.87), 0.005)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_multi_target()
