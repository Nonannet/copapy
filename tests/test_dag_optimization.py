import copapy as cp
from copapy._basic_types import value
from copapy.backend import get_dag_stats

def test_get_dag_stats():

    sum_size = 10
    v_size = 200

    v1 = cp.vector(cp.value(float(v)) for v in range(v_size))
    v2 = cp.vector(cp.value(float(v)) for v in [5]*v_size)

    v3 = sum((v1 + i + 7) @ v2 for i in range(sum_size))

    assert isinstance(v3, value)
    stat = get_dag_stats([v3])
    print(stat)

    assert stat['const_float'] == 2 * v_size
    assert stat['add_float_float'] == sum_size * v_size - 2


if __name__ == "__main__":
    test_get_dag_stats()