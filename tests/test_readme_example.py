import copapy as cp
import pytest

def test_readme_example():
    # Define variables
    a = cp.value(0.25)
    b = cp.value(0.87)

    # Define computations
    c = a + b * 2.0
    d = c ** 2 + cp.sin(a)
    e = cp.sqrt(b)

    # Create a target, compile and run
    tg = cp.Target()
    tg.compile(c, d, e)
    tg.run()

    # Read the results
    print("Result c:", tg.read_value(c))
    print("Result d:", tg.read_value(d))
    print("Result e:", tg.read_value(e))

    # Assertions to verify correctness
    assert tg.read_value(c) == pytest.approx(0.25 + 0.87 * 2.0, 0.001)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(d) == pytest.approx((0.25 + 0.87 * 2.0) ** 2 + cp.sin(0.25), 0.005)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(e) == pytest.approx(cp.sqrt(0.87), 0.001)  # pyright: ignore[reportUnknownMemberType]

if __name__ == "__main__":
    test_readme_example()
