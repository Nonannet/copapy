from copapy import variable, Target
import pytest
import copapy as cp


def test_corse():
    a_i = 9
    a_f = 2.5
    c_i = variable(a_i)
    c_f = variable(a_f)
    # c_b = variable(True)

    ret_test = (c_f ** c_f, c_i ** c_i) # , c_i & 3)
    ret_refe = (a_f ** a_f, a_i ** a_i) # , a_i & 3)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_refe):
        assert isinstance(test, cp.variable)
        val = tg.read_value(test)
        print('+', val, ref, type(val), test.dtype)
        #for t in (int, float, bool):
        #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 2), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.skip(reason="sqrt must be fixed")
def test_fine():
    a_i = 9
    a_f = 2.5
    c_i = variable(a_i)
    c_f = variable(a_f)
    # c_b = variable(True)

    ret_test = (c_f ** 2, c_i ** -1, cp.sqrt(c_i), cp.sqrt(c_f), cp.sin(c_f), cp.cos(c_f), cp.tan(c_f))  # , c_i & 3)
    ret_refe = (a_f ** 2, a_i ** -1, cp.sqrt(a_i), cp.sqrt(a_f), cp.sin(a_f), cp.cos(a_f), cp.tan(a_f))  # , a_i & 3)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_refe):
        assert isinstance(test, cp.variable)
        val = tg.read_value(test)
        print('+', val, ref, type(val), test.dtype)
        #for t in (int, float, bool):
        #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 0.001), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_corse()
    test_fine()
