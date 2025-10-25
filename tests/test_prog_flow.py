from copapy import variable, Target, iif
import pytest
import copapy


def test_compile():
    c_i = variable(9)
    c_f = variable(2.5)
    # c_b = variable(True)

    ret_test = (iif(c_f > 5, c_f, -1), iif(c_i > 5, c_f, 8.8), iif(c_i > 2, c_i, 1))
    ret_ref = (iif(2.5 > 5, 2.5, -1), iif(9 > 5, 2.5, 8.8), iif(9 > 2, 9, 1))

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_ref):
        assert isinstance(test, copapy.variable)
        val = tg.read_value(test)
        print('+', val, ref, type(val), test.dtype)
        #for t in (int, float, bool):
        #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 0.001), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_compile()
