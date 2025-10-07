from copapy import const, Target
from pytest import approx
import time

def function(c1, c2):
    i1 = c1 * 3.3 + 5
    i2 = c2 * 5 + c1
    r1 = i1 + i2 * 55 / 4
    r2 = 4 * i2 + 5

    return i1, i2, r1, r2 


def test_compile():

    c1 = const(4)
    c2 = const(2)
    
    ret = function(c1, c2)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret)
    #time.sleep(5)
    print('* run and copy ...')
    tg.run()
    #print('* finished')

    ret_ref = function(4, 2)

    for test, ref, name in zip(ret, ret_ref, ['i1', 'i2', 'r1', 'r2']):
        print('+', name)
        assert tg.read_value(test) == approx(ref, 1e-5), name


if __name__ == "__main__":
    test_compile()
