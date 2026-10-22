
import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# CSS314 TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD
# Student ID: 230103057
# Tasks: 10,000
# Workload: SHA-256 hashing of 1 KB data, 500 times per task
# ============================================================

STUDENT_ID = 230103057
TASK_COUNT = 10_000
HASH_ITERATIONS = 500
PAYLOAD_SIZE = 1024
REPEATS = 2


def cpu_bound_task(task_id):
    """Perform repeated SHA-256 hashing on a 1 KB payload."""
    payload = (
        f"StudentID={STUDENT_ID};Task={task_id};".encode()
    )
    payload = (payload * ((PAYLOAD_SIZE // len(payload)) + 1))[
        :PAYLOAD_SIZE
    ]

    result = b""

    for _ in range(HASH_ITERATIONS):
        result = hashlib.sha256(payload).digest()

    return result


def run_benchmark(pool_size):
    """Run all tasks and return wall-clock execution time in ms."""
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        # Consume every result so all tasks finish before timing ends.
        results = list(executor.map(cpu_bound_task, range(TASK_COUNT)))

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Validate that all task results were produced.
    assert len(results) == TASK_COUNT

    return elapsed_ms


def main():
    cores = os.cpu_count() or 1

    requested_sizes = [
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256,
    ]

    # Remove duplicates while preserving the requested order.
    pool_sizes = list(dict.fromkeys(requested_sizes))

    print("=" * 88)
    print("CSS314 TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU cores: {cores}")
    print(f"Tasks: {TASK_COUNT:,}")
    print(f"Hashes per task: {HASH_ITERATIONS}")
    print(f"Payload size: {PAYLOAD_SIZE} bytes")
    print(f"Repeats per pool size: {REPEATS} (median)")
    print("=" * 88)

    timings = {}

    for pool_size in pool_sizes:
        repeat_times = []

        for _ in range(REPEATS):
            elapsed_ms = run_benchmark(pool_size)
            repeat_times.append(elapsed_ms)

        timings[pool_size] = statistics.median(repeat_times)
        print(
            f"Finished pool size {pool_size}: "
            f"{timings[pool_size]:.2f} ms"
        )

    baseline = timings[1]

    print("\nRESULTS")
    print("-" * 88)
    print(
        f"{'Pool Size':>12} | {'Execution Time (ms)':>20} | "
        f"{'Speedup':>12} | {'Status':<30}"
    )
    print("-" * 88)

    previous_time = None
    first_slowdown_size = None
    best_time_so_far = float("inf")

    for pool_size in pool_sizes:
        elapsed_ms = timings[pool_size]
        speedup = baseline / elapsed_ms if elapsed_ms > 0 else float("nan")

        if pool_size == 1:
            status = "BASELINE"
        elif elapsed_ms > best_time_so_far:
            status = "SLOWER than best earlier"
        elif previous_time is not None and elapsed_ms > previous_time:
            status = "SLOWER than previous"
        else:
            status = "NO REGRESSION OBSERVED"

        # Mark the first observed worsening compared with any smaller
        # tested pool size. This is an empirical threshold, not proof
        # that context switching alone caused the slowdown.
        if (
            pool_size != 1
            and first_slowdown_size is None
            and elapsed_ms > best_time_so_far
        ):
            first_slowdown_size = pool_size
            status += "  <-- FIRST SLOWDOWN"

        print(
            f"{pool_size:>12} | {elapsed_ms:>20.2f} | "
            f"{speedup:>11.2f}x | {status:<30}"
        )

        best_time_so_far = min(best_time_so_far, elapsed_ms)
        previous_time = elapsed_ms

    print("-" * 88)

    if first_slowdown_size is None:
        print(
            "First slowdown: NOT OBSERVED relative to the best earlier "
            "tested pool size."
        )
    else:
        print(
            "First observed slowdown: pool size "
            f"{first_slowdown_size}."
        )

    print(
        "Note: timing alone cannot prove OS context switching is the "
        "only cause."
    )
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
