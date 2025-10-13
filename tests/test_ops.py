from copapy import CPVariable, Target
from pytest import approx


def function1(c1):
    return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
            c1 * 4, c1 * -4,
            c1 + 4, c1 - 4,
            c1 > 2, c1 > 100, c1 < 4, c1 < 100]

def function2(c1):
    return [c1 / 4.44, c1 / -4.44, c1 // 4.44, c1 // -4.44, (c1 * -1) // 4.44,
            c1 * 4.44, c1 * -4.44,
            c1 + 4.44, c1 - 4.44,
            c1 > 2, c1 > 100.11, c1 < 4.44, c1 < 100.11]

def function3(c1):
    return [c1 / 4]

def function4(c1):
    return [c1 == 9, c1 == 4, c1 != 9, c1 != 4]

def function5(c1):
    return [c1 == True, c1 == False, c1 != True, c1 != False]

def test_compile():

    c1 = CPVariable(9)
    c2 = CPVariable(1.111)
    c3 = CPVariable(False)

    #ret_test = function1(c1) + function1(c2) + function2(c1) + function2(c2) + function3(c3) + function4(c1) + function5(c3) + [CPVariable(9) % 2]
    #ret_ref = function1(9) + function1(1.111) + function2(9) + function2(1.111) + function3(9) + function4(9) + function5(True) + [9 % 2]

    ret_test = [c1 / 4]
    ret_ref =  [9 / 4]

    print(ret_test)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    #time.sleep(5)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_ref):
        val = tg.read_value(test)
        print('+', val, ref)
        for t in [int, float, bool]:
            assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == approx(ref, 1e-5), f"Result does not match: {val} and reference: {ref}"


if __name__ == "__main__":
    test_compile()
