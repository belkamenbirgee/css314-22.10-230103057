
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ============================================================
# CSS314 TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK
# Student ID: 230103057
# Total sample points: 40,000,000
# Threads: 4
# ============================================================

STUDENT_ID = 230103057
TOTAL_POINTS = 40_000_000
NUM_THREADS = 4
BATCH_SIZE = 250_000

# One generator shared by all threads in Mode A.
shared_rng = np.random.default_rng(STUDENT_ID)
rng_lock = threading.Lock()


def count_points_in_circle(x, y):
    """Count points inside the unit quarter-circle."""
    return int(np.count_nonzero(x * x + y * y <= 1.0))


def shared_prng_worker(points):
    """Mode A: generate random points using a shared locked PRNG."""
    inside = 0
    remaining = points

    while remaining > 0:
        batch = min(BATCH_SIZE, remaining)

        # Lock only while accessing the shared generator state.
        with rng_lock:
            x = shared_rng.random(batch)
            y = shared_rng.random(batch)

        # Circle calculation happens outside the lock.
        inside += count_points_in_circle(x, y)
        remaining -= batch

    return inside


def independent_prng_worker(args):
    """Mode B: each thread uses its own independently seeded PRNG."""
    thread_index, points = args

    seed = (STUDENT_ID * 10) + thread_index
    rng = np.random.default_rng(seed)

    inside = 0
    remaining = points

    while remaining > 0:
        batch = min(BATCH_SIZE, remaining)

        x = rng.random(batch)
        y = rng.random(batch)

        inside += count_points_in_circle(x, y)
        remaining -= batch

    return inside


def estimate_pi(inside, total):
    return 4.0 * inside / total


def main():
    points_per_thread = TOTAL_POINTS // NUM_THREADS

    print("=" * 62)
    print("CSS314 TASK 2: PRNG STATE BOTTLENECK")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total sample points: {TOTAL_POINTS:,}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Points per thread: {points_per_thread:,}")
    print("=" * 62)

    # --------------------------------------------------------
    # Mode A: shared PRNG with synchronization
    # --------------------------------------------------------
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(
            executor.map(
                shared_prng_worker,
                [points_per_thread] * NUM_THREADS
            )
        )

    shared_time_ms = (time.perf_counter() - start) * 1000
    shared_inside = sum(results)
    shared_pi = estimate_pi(shared_inside, TOTAL_POINTS)
    shared_error = abs(shared_pi - math.pi)

    # --------------------------------------------------------
    # Mode B: independent PRNG per thread
    # Seeds are (Student_ID * 10) + thread_index, 0 to 3.
    # --------------------------------------------------------
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(
            executor.map(
                independent_prng_worker,
                [(i, points_per_thread) for i in range(NUM_THREADS)]
            )
        )

    independent_time_ms = (time.perf_counter() - start) * 1000
    independent_inside = sum(results)
    independent_pi = estimate_pi(independent_inside, TOTAL_POINTS)
    independent_error = abs(independent_pi - math.pi)

    speedup = (
        shared_time_ms / independent_time_ms
        if independent_time_ms > 0 else float("nan")
    )

    print("\nRESULTS")
    print("-" * 62)

    print("Mode A — Shared PRNG with Lock")
    print(f"Computed pi:       {shared_pi:.10f}")
    print(f"Absolute error:    {shared_error:.10f}")
    print(f"Wall time:         {shared_time_ms:.2f} ms")

    print()

    print("Mode B — Independent PRNGs")
    print(f"Computed pi:       {independent_pi:.10f}")
    print(f"Absolute error:    {independent_error:.10f}")
    print(f"Wall time:         {independent_time_ms:.2f} ms")

    print()
    print(f"Speedup (A time / B time): {speedup:.2f}x")
    print("-" * 62)
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
