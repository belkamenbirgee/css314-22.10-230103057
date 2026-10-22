
import os
import time
import platform
import multiprocessing as mp
from multiprocessing import shared_memory

ITERATIONS = 50_000_000
CACHE_LINE = 64


def count_shared(args):
    """Increment one shared counter repeatedly."""
    index, iterations, shm_name = args
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        # Each counter is an unsigned 64-bit integer (8 bytes).
        counter = memoryview(shm.buf).cast("Q")
        for _ in range(iterations):
            counter[index] += 1
        counter.release()
    finally:
        shm.close()


def count_local_then_write(args):
    """Increment a local variable, then write once to shared memory."""
    index, iterations, shm_name = args
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        local_count = 0
        for _ in range(iterations):
            local_count += 1

        counter = memoryview(shm.buf).cast("Q")
        counter[index] = local_count
        counter.release()
    finally:
        shm.close()


def run_benchmark(worker_count, function, padded=False):
    stride = CACHE_LINE // 8 if padded else 1
    size = worker_count * stride * 8

    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        counters = memoryview(shm.buf).cast("Q")
        for i in range(worker_count * stride):
            counters[i] = 0
        counters.release()

        jobs = [
            (i * stride, ITERATIONS, shm.name)
            for i in range(worker_count)
        ]

        start = time.perf_counter()

        with mp.Pool(processes=worker_count) as pool:
            pool.map(function, jobs)

        elapsed_ms = (time.perf_counter() - start) * 1000

        counters = memoryview(shm.buf).cast("Q")
        values = [counters[i * stride] for i in range(worker_count)]
        counters.release()

        if any(value != ITERATIONS for value in values):
            raise RuntimeError("Counter verification failed!")

        return elapsed_ms
    finally:
        shm.close()
        shm.unlink()


def main():
    # Physical core count is not exposed directly by os.cpu_count().
    # This uses logical cores as a portable fallback.
    workers = os.cpu_count() or 1

    print("TASK 3 - FALSE SHARING PROFILE")
    print(f"Operating System: {platform.platform()}")
    print(f"Logical CPU cores: {workers}")
    print(f"Worker processes: {workers}")
    print(f"Increments per worker: {ITERATIONS:,}")
    print()

    unpadded_ms = run_benchmark(
        workers, count_shared, padded=False
    )
    local_ms = run_benchmark(
        workers, count_local_then_write, padded=True
    )

    print(f"{'Version':<25} {'Wall Time (ms)':>16}")
    print("-" * 43)
    print(f"{'Unpadded Shared':<25} {unpadded_ms:>16.3f}")
    print(f"{'Local + Padded':<25} {local_ms:>16.3f}")

    if local_ms > 0:
        print(
            "Execution Ratio (Unpadded / Local-Padded): "
            f"{unpadded_ms / local_ms:.3f}x"
        )

    print("Counter Verification: PASSED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
