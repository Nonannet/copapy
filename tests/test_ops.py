from copapy import CPVariable, Target
from pytest import approx


def function1(c1):
    return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]

def function2(c1):
    return [c1 / 4, c1 / -4, c1 / 4, c1 / -4, (c1 * -1) / 4]

def function3(c1):
    return [c1 / 4]

def test_compile():

    c1 = CPVariable(9)

    ret = function3(c1)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret)
    #time.sleep(5)
    print('* run and copy ...')
    tg.run()
    #print('* finished')

    ret_ref = function3(9)

    for test, ref, name in zip(ret, ret_ref, ['r1', 'r2', 'r3', 'r4', 'r5']):
        val = tg.read_value(test)
        print('+', name, val, ref)
        assert val == approx(ref, 1e-5), name


if __name__ == "__main__":
    test_compile()
