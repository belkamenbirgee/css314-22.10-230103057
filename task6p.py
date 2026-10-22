
import threading
import time

# ============================================================
# CSS314 TASK 6: SYNCHRONIZATION UNDER HEAVY CONTENTION
# Student ID: 230103057
# Threads: 8
# Increments per thread: 1,000,000
# Total operations: 8,000,000
# ============================================================

STUDENT_ID = 230103057
NUM_THREADS = 8
INCREMENTS_PER_THREAD = 1_000_000
TOTAL_OPERATIONS = NUM_THREADS * INCREMENTS_PER_THREAD


def mutex_worker(counter, lock):
    """Increment the shared counter using a blocking mutex."""
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            counter[0] += 1


def spinlock_worker(counter, lock):
    """
    Increment the shared counter using a busy-wait spinlock.

    Python does not expose a portable hardware CAS API.
    This is a spinlock-style demonstration, not a direct CAS instruction.
    """
    for _ in range(INCREMENTS_PER_THREAD):
        # Repeatedly attempt to acquire the lock without blocking.
        while not lock.acquire(blocking=False):
            pass

        try:
            counter[0] += 1
        finally:
            lock.release()


def run_benchmark(worker_function, label):
    """Run a synchronization benchmark and verify the final counter."""
    counter = [0]
    lock = threading.Lock()

    threads = [
        threading.Thread(
            target=worker_function,
            args=(counter, lock)
        )
        for _ in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_seconds = time.perf_counter() - start
    elapsed_ms = elapsed_seconds * 1000

    # Each increment must be accounted for.
    if counter[0] != TOTAL_OPERATIONS:
        raise RuntimeError(
            f"{label} failed: expected {TOTAL_OPERATIONS:,}, "
            f"got {counter[0]:,}"
        )

    throughput = (
        TOTAL_OPERATIONS / elapsed_seconds
        if elapsed_seconds > 0 else float("inf")
    )

    return elapsed_ms, throughput, counter[0]


def main():
    print("=" * 68)
    print("CSS314 TASK 6: MUTEX VS. SPINLOCK UNDER CONTENTION")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD:,}")
    print(f"Total operations per test: {TOTAL_OPERATIONS:,}")
    print("=" * 68)

    # Primitive 1: blocking mutex.
    mutex_ms, mutex_ops_sec, mutex_count = run_benchmark(
        mutex_worker,
        "Mutex"
    )

    print("\nPrimitive 1 — Mutex (threading.Lock)")
    print(f"Final counter:       {mutex_count:,}")
    print(f"Execution time:      {mutex_ms:.2f} ms")
    print(f"Throughput:          {mutex_ops_sec:,.0f} ops/sec")

    # Primitive 2: busy-wait spinlock.
    spin_ms, spin_ops_sec, spin_count = run_benchmark(
        spinlock_worker,
        "Spinlock"
    )

    print("\nPrimitive 2 — Spinlock (busy-wait)")
    print(f"Final counter:       {spin_count:,}")
    print(f"Execution time:      {spin_ms:.2f} ms")
    print(f"Throughput:          {spin_ops_sec:,.0f} ops/sec")

    ratio = (
        spin_ops_sec / mutex_ops_sec
        if mutex_ops_sec > 0 else float("nan")
    )

    print("\nCOMPARISON")
    print("-" * 68)
    print(f"Mutex Execution Time:       {mutex_ms:.2f} ms")
    print(f"Mutex Throughput:            {mutex_ops_sec:,.0f} ops/sec")
    print(f"Spinlock Execution Time:     {spin_ms:.2f} ms")
    print(f"Spinlock Throughput:          {spin_ops_sec:,.0f} ops/sec")
    print(f"Throughput Ratio (Spin/Mutex): {ratio:.3f}x")
    print("-" * 68)
    print("Verification: PASS — both counters reached the expected value.")
    print("Note: Python spinlock is not a direct hardware CAS benchmark.")
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
