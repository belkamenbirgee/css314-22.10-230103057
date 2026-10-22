
import os
import time
import statistics
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A = 2.5
B = 1.5
REPEATS = 3


def vector_operation(x, y, start, end):
    """Compute y[i] = A * x[i] + B for a disjoint slice."""
    np.multiply(x[start:end], A, out=y[start:end])
    np.add(y[start:end], B, out=y[start:end])


def serial_run(x, y):
    vector_operation(x, y, 0, len(x))


def parallel_run(x, y, workers):
    """Divide the vector into chunks and process them with threads."""
    n = len(x)
    chunk_size = (n + workers - 1) // workers
    ranges = [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]

    # Pool startup and shutdown are included in the measured time.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(vector_operation, x, y, start, end)
            for start, end in ranges
        ]
        for future in futures:
            future.result()


def median_time(run_function, x, workers=None):
    """Return median wall time over repeated runs."""
    measurements = []

    for _ in range(REPEATS):
        # Allocate output outside the timed region for both methods.
        y = np.empty_like(x)

        start = time.perf_counter()

        if workers is None:
            serial_run(x, y)
        else:
            parallel_run(x, y, workers)

        measurements.append(time.perf_counter() - start)

        # Verify the computed result.
        expected = A * x + B
        if not np.allclose(y, expected, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Numerical verification failed.")

    return statistics.median(measurements)


def main():
    max_cores = os.cpu_count() or 1

    print("TASK 5 - PARALLEL OVERHEAD CROSSOVER")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.uname().machine}")
    print(f"Logical CPU cores: {max_cores}")
    print(f"Parallel workers: {max_cores}")
    print(f"Repeats per measurement: {REPEATS}")
    print("Operation: y[i] = A * x[i] + B")
    print()

    results = []

    for n in SIZES:
        # Identical deterministic input for both implementations.
        x = np.linspace(0.0, 1.0, n, dtype=np.float64)

        # Warm-up NumPy before measuring.
        warmup_y = np.empty_like(x)
        serial_run(x, warmup_y)

        serial_seconds = median_time(x=x, run_function=None) \
            if False else median_time(serial_run, x)

        parallel_seconds = median_time(
            parallel_run, x, workers=max_cores
        )

        speedup = serial_seconds / parallel_seconds
        faster = parallel_seconds < serial_seconds

        results.append({
            "n": n,
            "serial": serial_seconds,
            "parallel": parallel_seconds,
            "speedup": speedup,
            "faster": faster
        })

    print(
        f"{'N':>12} | {'Serial (ms)':>12} | "
        f"{'Parallel (ms)':>14} | {'Speedup':>9} | "
        f"{'Winner':>10}"
    )
    print("-" * 72)

    for row in results:
        winner = "Parallel" if row["faster"] else "Serial"
        print(
            f"{row['n']:>12,} | "
            f"{row['serial'] * 1000:>12.3f} | "
            f"{row['parallel'] * 1000:>14.3f} | "
            f"{row['speedup']:>9.3f} | {winner:>10}"
        )

    print("\nCROSSOVER ANALYSIS")
    print("-" * 72)

    transitions = []

    for previous, current in zip(results, results[1:]):
        if not previous["faster"] and current["faster"]:
            transitions.append((previous["n"], current["n"]))

    if transitions:
        for lower, upper in transitions:
            print(
                f"Measured crossover interval: N > {lower:,} "
                f"and N <= {upper:,} (sampled interval)"
            )
        print(
            "Interpretation: the first sampled size faster in parallel "
            "is shown for each slower-to-faster transition."
        )
    elif all(not row["faster"] for row in results):
        print("Parallel execution was not faster at any tested size.")
        print(
            f"If a crossover exists, it is above {SIZES[-1]:,} "
            "or requires a different implementation."
        )
    elif all(row["faster"] for row in results):
        print("Parallel execution was faster at every tested size.")
        print(
            f"The crossover is at or below the smallest tested size "
            f"({SIZES[0]})."
        )
    else:
        print(
            "Performance alternated between serial and parallel. "
            "See the table for each measured size."
        )
        print(
            "No single monotonic crossover can be claimed from "
            "these measurements."
        )

    print("\nBenchmark completed. All numerical checks passed.")


if __name__ == "__main__":
    main()
