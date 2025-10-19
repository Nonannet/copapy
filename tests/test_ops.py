from copapy import cpvalue, Target, NumLike, Net, iif, cpint
from pytest import approx


def function1(c1: NumLike) -> list[NumLike]:
    return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
            c1 * 4, c1 * -4,
            c1 + 4, c1 - 4,
            c1 > 2, c1 > 100, c1 < 4, c1 < 100]


def function2(c1: NumLike) -> list[NumLike]:
    return [c1 / 4.44, c1 / -4.44, c1 // 4.44, c1 // -4.44, (c1 * -1) // 4.44,
            c1 * 4.44, c1 * -4.44,
            c1 + 4.44, c1 - 4.44,
            c1 > 2, c1 > 100.11, c1 < 4.44, c1 < 100.11]


def function3(c1: NumLike) -> list[NumLike]:
    return [c1 / 4]


def function4(c1: NumLike) -> list[NumLike]:
    return [c1 == 9, c1 == 4, c1 != 9, c1 != 4]


def function5(c1: NumLike) -> list[NumLike]:
    return [c1 == True, c1 == False, c1 != True, c1 != False, c1 / 2, c1 + 2]


def function6(c1: NumLike) -> list[NumLike]:
    return [c1 == True]


def iiftests(c1: NumLike) -> list[NumLike]:
    return [iif(c1 > 5, 8, 9),
            iif(c1 < 5, 8.5, 9.5),
            iif(1 > 5, 3.3, 8.8) + c1,
            iif(1 < 5, c1 * 3.3, 8.8),
            iif(c1 < 5, c1 * 3.3, 8.8)]


def test_compile():
    c_i = cpvalue(9)
    c_f = cpvalue(1.111)
    c_b = cpvalue(True)

    ret_test = function1(c_i) + function1(c_f) + function2(c_i) + function2(c_f) + function3(c_i) + function4(c_i) + function5(c_b) + [cpint(9) % 2] + iiftests(c_i) + iiftests(c_f)
    ret_ref = function1(9) + function1(1.111) + function2(9) + function2(1.111) + function3(9) + function4(9) + function5(True) + [9 % 2] + iiftests(9) + iiftests(1.111)

    tg = Target()
    print('* compile and copy ...')
    tg.compile(ret_test)
    print('* run and copy ...')
    tg.run()
    print('* finished')

    for test, ref in zip(ret_test, ret_ref):
        assert isinstance(test, Net)
        val = tg.read_value(test)
        print('+', val, ref, test.dtype)
        for t in [int, float, bool]:
            assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == approx(ref, 1e-5), f"Result does not match: {val} and reference: {ref}"


if __name__ == "__main__":
    test_compile()
