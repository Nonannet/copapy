from copapy import Write, const
import copapy as rc

def test_compile():
    c1 = const(1.11)
    c2 = const(2.22)

    i1 = c1 * 2
    i2 = i1 + 3

    r1 = i1 + i2
    r2 = c2 + 4 + c1

    out = [Write(r1), Write(r2)]

    il = rc.compile_to_instruction_list(out)

    print('#', il.print())


if __name__ == "__main__":
    test_compile()