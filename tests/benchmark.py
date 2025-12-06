import copapy as cp
import time
import json
import os
import subprocess
import sys
import numpy as np
from numpy.core._multiarray_umath import __cpu_features__

from copapy._matrices import diagonal

CPU_SIMD_FEATURES = "SSE SSE2 SSE3 SSSE3 SSE41 SSE42 AVX AVX2 AVX512F FMA3"


def cp_vs_python(path: str):
    os.environ.get("NPY_DISABLE_CPU_FEATURES")
    cpu_f = CPU_SIMD_FEATURES.split(' ')
    print('\n'.join(f"> {k}: {v}" for k, v in __cpu_features__.items() if k in cpu_f))


    results: list[dict[str, str | float | int]] = []

    for _ in range(15):
        for v_size in [10, 30, 60] + list(range(100, 600, 100)):

            sum_size = 10
            #v_size = 400
            iter_size = 30000

            v1 = cp.vector(cp.value(float(v)) for v in range(v_size))
            v2 = cp.vector(cp.value(float(v)) for v in [5]*v_size)

            v3 = sum((v1 + i) @ v2 for i in range(sum_size))

            tg = cp.Target()
            tg.compile(v3)

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size):
                tg.run()
            elapsed_cp = time.perf_counter() - t0

            #print(f"Copapy: {elapsed_cp:.4f} s")
            results.append({'benchmark': 'Copapy', 'iter_size': iter_size, 'elapsed_time': elapsed_cp, 'sum_size': sum_size, 'v_size': v_size})



            v1 = cp.vector(float(v) for v in range(v_size))
            v2 = cp.vector(float(v) for v in [5]*v_size)

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size//100):
                v3 = sum((v1 + i) @ v2 for i in range(sum_size))

            elapsed_python = time.perf_counter() - t0

            #print(f"Python: {elapsed_python:.4f} s")
            results.append({'benchmark': 'Python','iter_size': iter_size//10, 'elapsed_time': elapsed_python, 'sum_size': sum_size, 'v_size': v_size})

            v1 = np.array(list(range(v_size)), dtype=np.float32)
            v2 = np.array([5]*v_size, dtype=np.float32)
            i = np.array(list(range(sum_size)), dtype=np.int32).reshape([sum_size, 1])

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size):
                v3 = np.sum((v1 + i) @ v2)

            elapsed_np = time.perf_counter() - t0

            #print(f"Numpy 2: {elapsed_np2:.4f} s")
            results.append({'benchmark': 'NumPy', 'iter_size': iter_size, 'elapsed_time': elapsed_np, 'sum_size': sum_size, 'v_size': v_size})


            print(f"{v_size} {elapsed_cp}, {elapsed_python}, {elapsed_np}")

    with open(path, 'w') as f:
        json.dump(results, f)


def cp_vs_python_sparse(path: str = 'benchmark_results_001_sparse.json'):
    results: list[dict[str, str | float | int]] = []

    for _ in range(7):
        for v_size in [8, 8, 16, 20, 24, 32]:

            n_ones = int((v_size ** 2) * 0.5)
            n_zeros = (v_size ** 2) - n_ones
            mask = np.array([1] * n_ones + [0] * n_zeros).reshape((v_size, v_size))
            np.random.shuffle(mask)

            sum_size = 10
            #v_size = 400
            iter_size = 3000

            v1 = cp.vector(cp.value(float(v)) for v in range(v_size))
            v2 = cp.vector(cp.value(float(v)) for v in [5]*v_size)

            test = cp.vector(np.linspace(0, 1, v_size))

            assert False, test * v2

            v3 = sum(((cp.diagonal(v1) + i) * cp.matrix(mask)) @ v2 for i in range(sum_size))

            tg = cp.Target()
            tg.compile(v3)

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size):
                tg.run()
            elapsed_cp = time.perf_counter() - t0

            #print(f"Copapy: {elapsed_cp:.4f} s")
            results.append({'benchmark': 'Copapy', 'iter_size': iter_size, 'elapsed_time': elapsed_cp, 'sum_size': sum_size, 'v_size': v_size})



            v1 = cp.vector(float(v) for v in range(v_size))
            v2 = cp.vector(float(v) for v in [5]*v_size)

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size//1000):
                v3 = sum(((cp.diagonal(v1) + i) * cp.matrix(mask)) @ v2 for i in range(sum_size))

            elapsed_python = time.perf_counter() - t0

            #print(f"Python: {elapsed_python:.4f} s")
            results.append({'benchmark': 'Python','iter_size': iter_size//10, 'elapsed_time': elapsed_python, 'sum_size': sum_size, 'v_size': v_size})

            v1 = np.array(list(range(v_size)), dtype=np.float32)
            v2 = np.array([5]*v_size, dtype=np.float32)
            i_arr = np.array(list(range(sum_size)), dtype=np.int32).reshape([sum_size, 1, 1])
            tmp1 = v1 * np.eye(v_size) + i_arr

            time.sleep(0.1)
            t0 = time.perf_counter()
            for _ in range(iter_size):
                v3 = np.sum(((tmp1) * mask) @ v2)

            elapsed_np = time.perf_counter() - t0

            #print(f"Numpy 2: {elapsed_np2:.4f} s")
            results.append({'benchmark': 'NumPy', 'iter_size': iter_size, 'elapsed_time': elapsed_np, 'sum_size': sum_size, 'v_size': v_size})


            print(f"{v_size} {elapsed_cp}, {elapsed_python}, {elapsed_np}")

    with open(path, 'w') as f:
        json.dump(results, f)


def plot_results(path: str):
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # Load the benchmark results
    with open(path, 'r') as f:
        results = json.load(f)

    # Group data by benchmark and v_size, then calculate medians
    data_by_benchmark = defaultdict(lambda: defaultdict(list))

    for entry in results:
        benchmark = entry['benchmark']
        v_size = entry['v_size']
        elapsed_time = entry['elapsed_time']
        data_by_benchmark[benchmark][v_size].append(elapsed_time)

    # Calculate medians
    medians_by_benchmark = {}
    for benchmark, v_sizes in data_by_benchmark.items():
        medians_by_benchmark[benchmark] = {
            v_size: np.median(times)
            for v_size, times in v_sizes.items()
        }

    # Sort by v_size for plotting
    benchmarks = sorted(medians_by_benchmark.keys())
    v_sizes_set = sorted(set(v for benchmark_data in medians_by_benchmark.values() for v in benchmark_data.keys()))

    # Create the plot
    plt.figure(figsize=(10, 6))

    for benchmark in benchmarks:
        if benchmark != 'Python':
            v_sizes = sorted(medians_by_benchmark[benchmark].keys())
            elapsed_times = [medians_by_benchmark[benchmark][v] for v in v_sizes]
            plt.plot(v_sizes, elapsed_times, '.', label=benchmark)

    plt.xlabel('Vector Size (v_size)')
    plt.ylabel('Elapsed Time (seconds)')
    #plt.title('Benchmark Results: Elapsed Time vs Vector Size')
    plt.legend()
    #plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()

    # Save to PNG
    plt.savefig(path.replace('.json', '') + '.png', dpi=300)
    print("Plot saved")


if __name__ == "__main__":
    path1 = 'benchmark_results_001.json'
    path2 = 'benchmark_results_001_sparse.json'

    if 'no_simd' in sys.argv[1:]:
        os.environ["NPY_DISABLE_CPU_FEATURES"] = CPU_SIMD_FEATURES
        subprocess.run([sys.executable, "tests/benchmark.py"])
    elif 'plot' in sys.argv[1:]:
        plot_results(path1)
        #plot_results(path2)
    else:
        cp_vs_python(path1)
        plot_results(path1)

        #cp_vs_python_sparse(path2)
        #plot_results(path2)
