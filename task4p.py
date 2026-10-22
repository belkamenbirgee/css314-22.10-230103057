
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import numpy as np

# ============================================================
# CSS314 TASK 4: MEMORY TRANSFER / IPC OVERHEAD
# CPU-only: Pickle IPC vs Zero-copy Shared Memory
# Student ID: 230103057
# ============================================================

STUDENT_ID = 230103057
ARRAY_SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3


def receiver_worker(connection):
    """Receive arrays through a pipe or attach to shared memory."""
    while True:
        message = connection.recv()
        mode = message[0]

        if mode == "stop":
            break

        if mode == "pickle":
            array = message[1]
            checksum = float(np.sum(array))

        elif mode == "shared":
            _, shm_name, shape, dtype_str = message
            shm = shared_memory.SharedMemory(name=shm_name)

            try:
                array = np.ndarray(
                    shape,
                    dtype=np.dtype(dtype_str),
                    buffer=shm.buf
                )
                checksum = float(np.sum(array))
                del array
            finally:
                shm.close()

        else:
            raise ValueError(f"Unknown transfer mode: {mode}")

        connection.send(checksum)

    connection.close()


def benchmark_pickle(connection, array):
    """Path A: send the full array through a serialized IPC pipe."""
    start = time.perf_counter()

    connection.send(("pickle", array))
    checksum = connection.recv()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, checksum


def benchmark_shared_memory(connection, array):
    """Path B: copy into shared memory and send only its metadata."""
    shm = None
    start = time.perf_counter()

    try:
        shm = shared_memory.SharedMemory(
            create=True,
            size=array.nbytes
        )

        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf
        )
        shared_array[:] = array[:]

        connection.send(
            ("shared", shm.name, array.shape, array.dtype.str)
        )
        checksum = connection.recv()

        del shared_array

    finally:
        if shm is not None:
            shm.close()
            shm.unlink()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, checksum


def median_benchmark(function, connection, array):
    """Repeat the transfer and return its median time."""
    timings = []
    checksums = []

    for _ in range(REPEATS):
        elapsed_ms, checksum = function(connection, array)
        timings.append(elapsed_ms)
        checksums.append(checksum)

    return statistics.median(timings), checksums


def main():
    rng = np.random.default_rng(STUDENT_ID)

    print("=" * 76)
    print("CSS314 TASK 4: HOST-TO-PROCESS MEMORY TRANSFER PROFILE")
    print(f"Student ID: {STUDENT_ID}")
    print("Path A: Multiprocessing pipe with Pickle serialization")
    print("Path B: Shared memory with metadata-only IPC")
    print(f"Repeats per path: {REPEATS} (median reported)")
    print("=" * 76)

    # Spawn is compatible with Windows.
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()

    process = context.Process(
        target=receiver_worker,
        args=(child_connection,)
    )
    process.start()
    child_connection.close()

    try:
        print(
            f"\n{'N (elements)':>14} | "
            f"{'Array MiB':>10} | "
            f"{'Path A IPC (ms)':>16} | "
            f"{'Path B Shared (ms)':>19} | "
            f"{'A/B ratio':>10}"
        )
        print("-" * 82)

        for n in ARRAY_SIZES:
            # Generate input outside the timed transfer.
            array = rng.random(n, dtype=np.float64)

            pickle_ms, pickle_checksums = median_benchmark(
                benchmark_pickle, parent_connection, array
            )

            shared_ms, shared_checksums = median_benchmark(
                benchmark_shared_memory, parent_connection, array
            )

            # Validate that both paths transferred the same data.
            expected = float(np.sum(array))

            if not all(
                np.isclose(value, expected, rtol=1e-12, atol=1e-8)
                for value in pickle_checksums + shared_checksums
            ):
                raise RuntimeError(
                    f"Transfer verification failed for N={n}"
                )

            ratio = pickle_ms / shared_ms if shared_ms > 0 else float("nan")
            size_mib = array.nbytes / (1024 ** 2)

            print(
                f"{n:>14,} | "
                f"{size_mib:>10.2f} | "
                f"{pickle_ms:>16.2f} | "
                f"{shared_ms:>19.2f} | "
                f"{ratio:>9.2f}x"
            )

            del array

        print("-" * 82)
        print("Verification: PASS — both paths produced matching checksums.")
        print("Timing includes transfer, receiver checksum, and")
        print("shared-memory allocation/copy/cleanup for Path B.")
        print("Process startup and random-array generation are excluded.")
        print("Benchmark completed.")

    finally:
        try:
            parent_connection.send(("stop",))
        except (BrokenPipeError, EOFError, OSError):
            pass

        parent_connection.close()
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
