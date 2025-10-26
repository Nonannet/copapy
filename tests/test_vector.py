import copapy as cp

def test_vectors_init():
    tt1 = cp.vector(range(3)) + cp.vector([1.1,2.2,3.3])
    tt2 = cp.vector([1.1,2,cp.variable(5)])# + cp.vector(range(3))
    tt3 = (cp.vector(range(3)) + 5.6)
    tt4 = cp.vector([1.1,2,3]) + cp.vector(cp.variable(v) for v in range(3))
    tt5 = cp.vector([1,2,3]).dot(tt4)

    print(tt1, tt2, tt3, tt4, tt5)


def test_compiled_vectors():
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.variable(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.variable(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).magnitude()
    #t4 = ((t3 * t1) * 2).sum()
    t5 = cp.sqrt(cp.variable(8.0))

    tg = cp.Target()
    tg.compile(t2, t4, t5)
    tg.run()

    assert isinstance(t2, cp.variable) and tg.read_value(t2) == 10 + 11 + 12 + 0 + 1 + 2
    #assert isinstance(t4, cp.variable) and tg.read_value(t4) == ((1/1*10 + 1/2*11 + 1/3*12) * 2)**0.5
    assert isinstance(t5, cp.variable) and tg.read_value(t5) == 8.0 * 3.5 + 4.5

if __name__ == "__main__":
    test_compiled_vectors()
