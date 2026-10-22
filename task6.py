
import threading
import time
import platform

NUM_THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = NUM_THREADS * INCREMENTS_PER_THREAD

# Shared state
shared_counter = 0
locked_counter = 0
counter_lock = threading.Lock()


def unsafe_increment():
    """Intentionally unsafe read-modify-write operation."""
    global shared_counter

    for _ in range(INCREMENTS_PER_THREAD):
        old_value = shared_counter
        new_value = old_value + 1
        shared_counter = new_value


def safe_increment():
    """Correct increment protected by a mutex lock."""
    global locked_counter

    for _ in range(INCREMENTS_PER_THREAD):
        with counter_lock:
            locked_counter += 1


def run_test(worker_function):
    threads = [
        threading.Thread(target=worker_function)
        for _ in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return (time.perf_counter() - start) * 1000


def main():
    global shared_counter, locked_counter

    print("TASK 6 - SYNCHRONIZATION COST & DATA LOSS")
    print(f"Operating System: {platform.platform()}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD:,}")
    print(f"Expected final value: {EXPECTED:,}")
    print()

    # Test 1: No synchronization
    shared_counter = 0
    unsafe_time_ms = run_test(unsafe_increment)

    lost_updates = EXPECTED - shared_counter
    lost_percentage = (lost_updates / EXPECTED) * 100

    print("UNSYNCHRONIZED VERSION")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {shared_counter:,}")
    print(f"Lost updates:         {lost_updates:,}")
    print(f"Updates lost:         {lost_percentage:.4f}%")
    print(f"Wall-clock time:      {unsafe_time_ms:.3f} ms")
    print()

    # Test 2: Correct synchronization
    locked_counter = 0
    safe_time_ms = run_test(safe_increment)

    print("LOCKED VERSION")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {locked_counter:,}")
    print(f"Updates lost:         {EXPECTED - locked_counter:,}")
    print(f"Wall-clock time:      {safe_time_ms:.3f} ms")

    print()
    print("SYNCHRONIZATION OVERHEAD")
    print(f"Additional time:      {safe_time_ms - unsafe_time_ms:.3f} ms")

    if unsafe_time_ms > 0:
        print(
            f"Time ratio (locked / unsafe): "
            f"{safe_time_ms / unsafe_time_ms:.3f}x"
        )

    print()
    print(
        "Verification:",
        "PASSED" if locked_counter == EXPECTED else "FAILED"
    )


if __name__ == "__main__":
    main()
