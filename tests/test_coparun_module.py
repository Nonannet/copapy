from copapy import const, Target
from pytest import approx


def function(c1):
    r1 = c1 / 2

    return [r1]


def test_compile():

    c1 = const(16)

    ret = function(c1)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret)
    #time.sleep(5)
    print('* run and copy ...')
    tg.run()
    #print('* finished')

    ret_ref = function(16)

    for test, ref, name in zip(ret, ret_ref, ['i1', 'i2', 'r1', 'r2']):
        val = tg.read_value(test)
        print('+', name, val, ref)
        assert val == approx(ref, 1e-5), name


if __name__ == "__main__":
    test_compile()
