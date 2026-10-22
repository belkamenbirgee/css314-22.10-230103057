
import time
import threading
import queue
import platform

NUM_ITEMS = 1000
NUM_WORKERS = 4
CHUNK_SIZE = 5


def process_item(i):
    """Simulate a non-uniform workload."""
    time.sleep(0.0001 * (i % 20))


def run_static():
    """Give each worker one fixed block of 250 items."""
    completion_times = [0.0] * NUM_WORKERS
    start_time = time.perf_counter()

    def worker(worker_id, items):
        for i in items:
            process_item(i)
        completion_times[worker_id] = time.perf_counter()

    threads = []
    block_size = NUM_ITEMS // NUM_WORKERS

    for worker_id in range(NUM_WORKERS):
        start = worker_id * block_size
        end = start + block_size
        thread = threading.Thread(
            target=worker,
            args=(worker_id, range(start, end))
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    return total_time, completion_times


def run_dynamic():
    """Workers repeatedly pull chunks from a shared queue."""
    work_queue = queue.Queue()
    completion_times = [0.0] * NUM_WORKERS

    # Pre-fill the queue with chunks of 5 items.
    for start in range(0, NUM_ITEMS, CHUNK_SIZE):
        work_queue.put(range(start, min(start + CHUNK_SIZE, NUM_ITEMS)))

    start_time = time.perf_counter()

    def worker(worker_id):
        while True:
            try:
                items = work_queue.get_nowait()
            except queue.Empty:
                break

            try:
                for i in items:
                    process_item(i)
            finally:
                work_queue.task_done()

        completion_times[worker_id] = time.perf_counter()

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(NUM_WORKERS)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    return total_time, completion_times


def report(name, total_time, completion_times):
    earliest = min(completion_times)
    latest = max(completion_times)

    # Completion timestamps are measured relative to the run start.
    # This function receives absolute timestamps, so their difference
    # gives the time between the first and last worker finishing.
    completion_delta = latest - earliest

    print(f"\n{name}")
    print("-" * 45)
    print(f"Total execution time: {total_time:.6f} seconds")
    print(f"First worker completion: {(earliest - run_start):.6f} seconds")
    print(f"Last worker completion:  {(latest - run_start):.6f} seconds")
    print(f"Min/max completion delta: {completion_delta:.6f} seconds")


if __name__ == "__main__":
    print("TASK 4 - WORK SCHEDULING UNDER SKEW")
    print(f"Operating System: {platform.platform()}")
    print(f"Items: {NUM_ITEMS}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Dynamic chunk size: {CHUNK_SIZE}")
    print("Workload: sleep(0.0001 * (i % 20))")

    # Static run
    run_start = time.perf_counter()
    static_time, static_completions = run_static()
    report("Strategy A: Static Scheduling",
           static_time, static_completions)

    # Dynamic run
    run_start = time.perf_counter()
    dynamic_time, dynamic_completions = run_dynamic()
    report("Strategy B: Dynamic Scheduling",
           dynamic_time, dynamic_completions)

    print("\nComparison")
    print("-" * 45)
    print(f"Static total time:  {static_time:.6f} seconds")
    print(f"Dynamic total time: {dynamic_time:.6f} seconds")

    if dynamic_time > 0:
        print(f"Static / Dynamic ratio: {static_time / dynamic_time:.3f}x")

    print("\nCompleted: both strategies processed 1000 items.")
