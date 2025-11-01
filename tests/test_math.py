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


def test_fine():
    a_i = 9
    a_f = 2.5
    c_i = variable(a_i)
    c_f = variable(a_f)
    # c_b = variable(True)

    ret_test = (c_f ** 2,
                c_i ** -1,
                cp.sqrt(c_i),
                cp.sqrt(c_f),
                cp.sin(c_f),
                cp.cos(c_f),
                cp.tan(c_f))  # , c_i & 3)
    
    ret_refe = (a_f ** 2,
                a_i ** -1,
                cp.sqrt(a_i),
                cp.sqrt(a_f),
                cp.sin(a_f),
                cp.cos(a_f),
                cp.tan(a_f))  # , a_i & 3)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref, name in zip(ret_test, ret_refe, ('^2', '**-1', 'sqrt_int', 'sqrt_float', 'sin', 'cos', 'tan')):
        assert isinstance(test, cp.variable)
        val = tg.read_value(test)
        print('+', val, ref, type(val), test.dtype)
        #for t in (int, float, bool):
        #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 1e-5), f"Result  for {name} does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


def test_trig_precision():

    test_vals = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0]  # up to 2pi

    ret_test = [r for v in test_vals for r in (cp.sin(variable(v)), cp.cos(variable(v)), cp.tan(variable(v)))]
    ret_refe = [r for v in test_vals for r in (cp.sin(v), cp.cos(v), cp.tan(v))]


    tg = Target()
    tg.compile(ret_test)
    tg.run()

    for i, (test, ref) in enumerate(zip(ret_test, ret_refe)):
        func_name = ['sin', 'cos', 'tan'][i % 3]
        assert isinstance(test, cp.variable)
        val = tg.read_value(test)
        print(f"+ Result of {func_name}: {val}; reference: {ref}")
        assert val == pytest.approx(ref, abs=1e-5), f"Result of {func_name} for input {test_vals[i // 3]} does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_corse()
    test_fine()
