
import os
import platform
import time
import numpy as np

# Student information
STUDENT_ID = 230103057
ARRAY_SIZE = 20_000_000

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {platform.processor() or platform.uname().machine}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    # Start timing the complete baseline workload
    start_time = time.perf_counter()

    # Generate 20 million float64 values using the Student ID as seed
    rng = np.random.default_rng(seed=STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    # Serial calculations: one NumPy reduction at a time
    total_sum = np.sum(data, dtype=np.float64)
    standard_deviation = np.std(data, dtype=np.float64, ddof=0)

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"Array Size: {ARRAY_SIZE}")
    print(f"Data Type: float64")
    print(f"Random Seed: {STUDENT_ID}")
    print(f"Mathematical Sum: {total_sum:.10f}")
    print(f"Population Standard Deviation: {standard_deviation:.10f}")
    print(f"Execution Time: {elapsed_ms:.3f} ms")

if __name__ == "__main__":
    main()
