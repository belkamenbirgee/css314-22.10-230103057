
import os
import platform
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103057
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95
REPEATS = 3


def parallel_sum(data, workers):
    """Sum chunks of the array using the requested number of threads."""
    chunks = np.array_split(data, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = list(executor.map(np.sum, chunks))

    return sum(partial_sums)


def main():
    max_cores = os.cpu_count() or 1

    # Keep the requested worker counts, removing duplicates if any.
    worker_counts = sorted(set([
        1, 2, 4, max_cores, max_cores * 2
    ]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {platform.processor() or platform.uname().machine}")
    print(f"Logical CPU Cores: {max_cores}")
    print(f"Array Size: {ARRAY_SIZE:,}")
    print("Data Type: float64")
    print(f"Random Seed: {STUDENT_ID}")
    print(f"Parallelizable Fraction (P): {PARALLEL_FRACTION}")
    print(f"Benchmark Repeats: {REPEATS}")

    # Generate data once, outside the timed benchmark.
    rng = np.random.default_rng(seed=STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    results = []

    for workers in worker_counts:
        # Warm-up to reduce first-run effects.
        parallel_sum(data, workers)

        times = []
        result_sum = None

        for _ in range(REPEATS):
            start = time.perf_counter()
            result_sum = parallel_sum(data, workers)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Median is more robust against background system activity.
        wall_time = statistics.median(times)
        results.append({
            "workers": workers,
            "time": wall_time,
            "sum": result_sum
        })

    # T(1) is the measured one-worker baseline.
    baseline_time = next(
        row["time"] for row in results if row["workers"] == 1
    )

    print()
    print(
        f"{'Cores':>8} | {'Wall Time (ms)':>15} | "
        f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}"
    )
    print("-" * 63)

    for row in results:
        workers = row["workers"]
        wall_time = row["time"]
        speedup = baseline_time / wall_time

        # Amdahl's Law: S(p) = 1 / ((1-P) + P/p)
        amdahl_bound = 1.0 / (
            (1.0 - PARALLEL_FRACTION)
            + PARALLEL_FRACTION / workers
        )

        print(
            f"{workers:>8} | {wall_time * 1000:>15.3f} | "
            f"{speedup:>10.3f} | {amdahl_bound:>19.3f}"
        )

    print()
    print(f"Serial baseline T(1): {baseline_time * 1000:.3f} ms")
    print(f"Verified sum (1 worker): "
          f"{next(r['sum'] for r in results if r['workers'] == 1):.10f}")


if __name__ == "__main__":
    main()
