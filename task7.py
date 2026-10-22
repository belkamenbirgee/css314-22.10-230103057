
import os
import time
import platform
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103057
ARRAY_SIZE = 30_000_000
REPEATS = 3


def make_chunks(n, workers):
    chunk_size = (n + workers - 1) // workers
    return [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]


def copy_kernel(a, b, start, end):
    np.copyto(b[start:end], a[start:end])


def compute_kernel(a, b, start, end):
    x = a[start:end]
    y = b[start:end]

    # Compute y = sin(x) * cos(x) + tan(x), using reusable buffers.
    np.sin(x, out=y)
    np.cos(x, out=x_temp_view(x, start, end))
    temp = x_temp_view(x, start, end)
    np.multiply(y, temp, out=y)

    np.tan(x, out=temp)
    np.add(y, temp, out=y)


# Reusable per-worker temporary arrays are allocated separately below.
def x_temp_view(x, start, end):
    return TEMP_BUFFER[start:end]


TEMP_BUFFER = None


def run_kernel(kernel, a, b, workers, executor=None):
    chunks = make_chunks(len(a), workers)

    if workers == 1:
        kernel(a, b, 0, len(a))
    else:
        futures = [
            executor.submit(kernel, a, b, start, end)
            for start, end in chunks
        ]
        for future in futures:
            future.result()


def benchmark(kernel, a, b, workers):
    global TEMP_BUFFER

    # Temporary storage is only needed by the compute kernel.
    if kernel == compute_kernel:
        TEMP_BUFFER = np.empty_like(a)

    # Warm-up before timing.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        run_kernel(kernel, a, b, workers, executor)

        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            run_kernel(kernel, a, b, workers, executor)
            times.append(time.perf_counter() - start)

    return statistics.median(times)


def report_kernel(name, kernel, a, b, worker_counts):
    print(f"\n{name}")
    print("-" * 65)
    print(f"{'Cores':>8} | {'Wall Time (s)':>14} | {'Speedup':>10} | {'Efficiency':>12}")

    results = {}

    for workers in worker_counts:
        elapsed = benchmark(kernel, a, b, workers)
        results[workers] = elapsed

    baseline = results[1]

    for workers in worker_counts:
        elapsed = results[workers]
        speedup = baseline / elapsed
        efficiency = speedup / workers

        print(
            f"{workers:>8} | {elapsed:>14.4f} | "
            f"{speedup:>10.3f} | {efficiency:>11.1%}"
        )

    return results


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(set([1, 2, max_cores]))

    print("TASK 7 - ARITHMETIC INTENSITY VS MEMORY BOTTLENECKS")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.uname().machine}")
    print(f"Logical CPU cores: {max_cores}")
    print(f"Array size: {ARRAY_SIZE:,} float64 elements")
    print(f"Benchmark repetitions: {REPEATS}")

    rng = np.random.default_rng(STUDENT_ID)

    # Keep tangent values away from its singularities.
    a = rng.uniform(-0.5, 0.5, ARRAY_SIZE).astype(np.float64)
    b = np.empty_like(a)

    copy_results = report_kernel(
        "Kernel A: Memory-Bound Vector Copy",
        copy_kernel, a, b, worker_counts
    )

    compute_results = report_kernel(
        "Kernel B: Compute-Bound Trigonometric Operation",
        compute_kernel, a, b, worker_counts
    )

    # Interpret scaling relative to the one-worker baseline.
    max_copy_speedup = copy_results[1] / copy_results[max_cores]
    max_compute_speedup = compute_results[1] / compute_results[max_cores]

    print("\nSCALING ANALYSIS")
    print("-" * 65)
    print(f"Copy speedup at {max_cores} cores: {max_copy_speedup:.3f}x")
    print(f"Compute speedup at {max_cores} cores: {max_compute_speedup:.3f}x")

    if max_cores > 1 and max_copy_speedup < max_compute_speedup:
        print(
            "Conclusion: Kernel A (vector copy) shows weaker scaling "
            "than Kernel B, consistent with a memory-bandwidth bottleneck."
        )
        print(
            "Memory-channel saturation is a plausible explanation, "
            "but timing alone does not prove saturation."
        )
    else:
        print(
            "Conclusion: The measured results do not show weaker scaling "
            "for Kernel A than Kernel B."
        )
        print(
            "Memory-channel saturation was not established by this run."
        )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
