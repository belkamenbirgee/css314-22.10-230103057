
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ============================================================
# CSS314 TASK 3: PREFIX SCAN
# Step Efficiency vs. Work Efficiency
# Student ID: 230103057
# ============================================================

STUDENT_ID = 230103057
ARRAY_SIZE = 2_000_000
CHUNK_COUNT = min(os.cpu_count() or 1, ARRAY_SIZE)


def serial_prefix_sum(data):
    """Serial inclusive prefix sum."""
    output = np.empty(len(data), dtype=np.int64)

    if len(data) == 0:
        return output

    output[0] = data[0]

    for i in range(1, len(data)):
        output[i] = output[i - 1] + data[i]

    return output


def calculate_local_scan(args):
    """Pass 1: compute a chunk's local prefix sums and total."""
    chunk_index, data, start, end = args

    local_prefix = np.cumsum(data[start:end], dtype=np.int64)
    local_total = int(local_prefix[-1]) if len(local_prefix) else 0

    return chunk_index, local_prefix, local_total


def apply_offset(args):
    """Pass 3: add the preceding chunks' total to a local scan."""
    local_prefix, offset = args

    if offset:
        local_prefix += offset

    return local_prefix


def parallel_prefix_sum(data, workers):
    """Two-pass parallel scan: local scans, offsets, then adjustment."""
    boundaries = [
        (i * len(data) // workers, (i + 1) * len(data) // workers)
        for i in range(workers)
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Pass 1: calculate local prefix sums and chunk totals.
        tasks = [
            (i, data, start, end)
            for i, (start, end) in enumerate(boundaries)
        ]
        local_results = list(executor.map(calculate_local_scan, tasks))

        # Keep chunks in their original order.
        local_results.sort(key=lambda result: result[0])

        local_prefixes = [result[1] for result in local_results]
        chunk_totals = [result[2] for result in local_results]

        # Pass 2: calculate each chunk's exclusive prefix offset.
        offsets = []
        running_total = 0

        for total in chunk_totals:
            offsets.append(running_total)
            running_total += total

        # Apply offsets to each chunk in parallel.
        adjusted_chunks = list(
            executor.map(
                apply_offset,
                zip(local_prefixes, offsets)
            )
        )

    return np.concatenate(adjusted_chunks)


def main():
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    print("=" * 62)
    print("CSS314 TASK 3: PARALLEL PREFIX SCAN")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Array size: {ARRAY_SIZE:,}")
    print(f"Available logical cores: {os.cpu_count() or 1}")
    print(f"Parallel workers: {CHUNK_COUNT}")
    print("=" * 62)

    # Benchmark the serial inclusive prefix sum.
    start = time.perf_counter()
    serial_output = serial_prefix_sum(data)
    serial_time_ms = (time.perf_counter() - start) * 1000

    # Benchmark the parallel chunked prefix scan.
    start = time.perf_counter()
    parallel_output = parallel_prefix_sum(data, CHUNK_COUNT)
    parallel_time_ms = (time.perf_counter() - start) * 1000

    speedup = (
        serial_time_ms / parallel_time_ms
        if parallel_time_ms > 0 else float("nan")
    )

    # Required verification: exit with error code 1 if it fails.
    try:
        assert parallel_output[-1] == serial_output[-1], (
            f"Final sum mismatch: parallel={parallel_output[-1]}, "
            f"serial={serial_output[-1]}"
        )
    except AssertionError as error:
        print("\nVerification Status: FAIL")
        print(f"ERROR: {error}")
        raise SystemExit(1)

    print("\nRESULTS")
    print("-" * 62)
    print(f"Serial Time:       {serial_time_ms:.2f} ms")
    print(f"Parallel Time:     {parallel_time_ms:.2f} ms")
    print(f"Speedup:           {speedup:.2f}x")
    print("Verification Status: PASS")
    print(f"Serial final sum:   {serial_output[-1]}")
    print(f"Parallel final sum: {parallel_output[-1]}")
    print("-" * 62)
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
