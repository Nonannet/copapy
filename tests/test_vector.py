import copapy as cp

def test_vec():
    tt = cp.vector(range(3)) + cp.vector([1.1,2.2,3.3])
    tt2 = (cp.vector(range(3)) + 5.6)