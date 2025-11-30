import copapy as cp
import numpy as np
import time

def cp_vs_python():

    from numpy.core._multiarray_umath import __cpu_features__
    print(__cpu_features__)

    for v_size in range(10, 800, 40):

        sum_size = 10
        #v_size = 400
        iter_size = 30000

        v1 = cp.vector(cp.variable(float(v)) for v in range(v_size))
        v2 = cp.vector(cp.variable(float(v)) for v in [5]*v_size)

        v3 = sum((v1 + i) @ v2 for i in range(sum_size))

        tg = cp.Target()
        tg.compile(v3)

        time.sleep(0.1)
        t0 = time.perf_counter()
        for _ in range(iter_size):
            tg.run()
        elapsed_cp = time.perf_counter() - t0

        #print(f"Copapy: {elapsed_cp:.4f} s")


        v1 = cp.vector(float(v) for v in range(v_size))
        v2 = cp.vector(float(v) for v in [5]*v_size)

        time.sleep(0.1)
        t0 = time.perf_counter()
        for _ in range(iter_size//10):
            v3 = sum((v1 + i) @ v2 for i in range(sum_size))

        elapsed_python = time.perf_counter() - t0

        #print(f"Python: {elapsed_python:.4f} s")


        i = np.array(list(range(sum_size)),).reshape([sum_size, 1])

        time.sleep(0.1)
        t0 = time.perf_counter()
        for _ in range(iter_size):
            v3 = np.sum((v1 + i) @ v2)

        elapsed_np2 = time.perf_counter() - t0

        #print(f"Numpy 2: {elapsed_np2:.4f} s")


        print(f"{elapsed_cp}, {elapsed_python}, {elapsed_np2}")

if __name__ == "__main__":
    cp_vs_python()