
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

# ============================================================
# CSS314 TASK 1: ROW-MAJOR VS COLUMN-MAJOR CACHE TRAVERSAL
# Student ID: 230103057
# Matrix: 8000 x 8000 float64 = approximately 512 MB
# ============================================================

STUDENT_ID = 230103057
N = 8000

_MATRIX = None
_SHARED_MEMORY = None


def initialize_worker(shm_name, shape):
    """Attach each worker to the shared matrix without copying it."""
    global _MATRIX, _SHARED_MEMORY

    _SHARED_MEMORY = shared_memory.SharedMemory(name=shm_name)
    _MATRIX = np.ndarray(
        shape,
        dtype=np.float64,
        buffer=_SHARED_MEMORY.buf
    )


def row_major_worker(bounds):
    """Mode A: each worker processes contiguous rows."""
    start, end = bounds
    return float(np.sum(_MATRIX[start:end, :]))


def column_major_worker(bounds):
    """Mode B: each worker processes columns, one column at a time."""
    start, end = bounds
    total = 0.0

    # Each column is strided in a C-order (row-major) matrix.
    for col in range(start, end):
        total += float(np.sum(_MATRIX[:, col]))

    return total


def make_chunks(length, workers):
    """Split the rows or columns into balanced worker ranges."""
    return [
        (i * length // workers, (i + 1) * length // workers)
        for i in range(workers)
    ]


def main():
    logical_cores = os.cpu_count() or 1

    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        physical_cores = None

    if not physical_cores:
        print("Physical core count unavailable; using logical core count.")
        physical_cores = logical_cores

    workers = min(physical_cores, N)

    print("=" * 58)
    print("CSS314 TASK 1: SPATIAL CACHE PROFILING")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {logical_cores}")
    print(f"Physical cores used: {workers}")
    print(f"Matrix size: {N} x {N}, float64")
    print(f"Matrix memory: {N * N * 8 / (1024 ** 2):.2f} MiB")
    print("=" * 58)

    shm = None

    try:
        # Allocate shared memory so workers can access the same matrix.
        shm = shared_memory.SharedMemory(
            create=True,
            size=N * N * np.dtype(np.float64).itemsize
        )
        matrix = np.ndarray(
            (N, N),
            dtype=np.float64,
            buffer=shm.buf
        )

        # Deterministic matrix initialization using the Student ID seed.
        rng = np.random.default_rng(STUDENT_ID)
        rng.random((N, N), dtype=np.float64, out=matrix)

        row_chunks = make_chunks(N, workers)
        column_chunks = make_chunks(N, workers)

        # Reuse one process pool for a fairer traversal comparison.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(shm.name, (N, N))
        ) as executor:

            # Mode A: row-major / contiguous traversal.
            start = time.perf_counter()
            row_results = list(
                executor.map(row_major_worker, row_chunks)
            )
            row_time_ms = (time.perf_counter() - start) * 1000

            # Mode B: column-wise / strided traversal.
            start = time.perf_counter()
            column_results = list(
                executor.map(column_major_worker, column_chunks)
            )
            column_time_ms = (time.perf_counter() - start) * 1000

        row_sum = sum(row_results)
        column_sum = sum(column_results)
        penalty = (
            column_time_ms / row_time_ms
            if row_time_ms > 0 else float("nan")
        )

        print("\nRESULTS")
        print("-" * 58)
        print(f"Mode A (Row-Major):       {row_time_ms:.2f} ms")
        print(f"Mode B (Column-Major):    {column_time_ms:.2f} ms")
        print(f"Cache penalty factor:     {penalty:.2f}x")
        print(f"Row-major sum:            {row_sum:.6f}")
        print(f"Column-major sum:         {column_sum:.6f}")
        print("-" * 58)
        print("Benchmark completed.")

    finally:
        if shm is not None:
            # Ensure the parent matrix view is released before closing.
            if "matrix" in locals():
                del matrix
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
