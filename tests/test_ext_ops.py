from copapy import variable, Target
import pytest
import copapy


def test_compile():
    c_i = variable(9)
    c_f = variable(2.5)
    # c_b = variable(True)

    ret_test = (c_f ** c_f, c_i ** c_i)#, c_i & 3)
    ret_ref = (2.5 ** 2.5, 9 ** 9)#, 9 & 3)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_ref):
        assert isinstance(test, copapy.CPNumber)
        val = tg.read_value(test)
        print('+', val, ref, type(val), test.dtype)
        #for t in (int, float, bool):
        #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 2), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_compile()
