from coparun_module import coparun
from copapy import Write, const, Target
import copapy
from copapy import binwrite


def test_compile():

    c1 = const(4)
    c2 = const(2) * 4

    i1 = c2 * 2
    r1 = i1 + 7 + (c1 + 7 * 9)
    r2 = i1 + 9

    tg = Target()
    tg.compile(r1, r2, c2)
    tg.run()

    print(tg.read_value(r1))
    print(tg.read_value(r2))
    print(tg.read_value(c2))


if __name__ == "__main__":
    test_compile()
