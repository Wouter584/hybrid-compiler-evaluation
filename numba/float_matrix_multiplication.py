import time
import numpy as np
from numba import cuda

@cuda.jit
def matrix_mul(a, b, c, n):
    tid = cuda.grid(1)
    i = tid // n
    j = tid % n
    if i < n and j < n:
        sum = 0
        for k in range(n):
            sum += a[i * n + k] * b[k * n + j]
        c[i * n + j] = sum

def bench9():
    # 1 is for precompilation
    sizes = [1, 16, 64, 256, 512, 1024, 2048, 4096]
    results = []

    for n in sizes:
        total = n * n

        a = np.array([(i * 12345.6789) % 1000 for i in range(total)], dtype=float)
        b = np.array([(i * 98765.4321) % 1000 for i in range(total)], dtype=float)
        c = np.zeros(total, dtype=float)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        threads_per_block = 256
        blocks = (total + threads_per_block - 1) // threads_per_block

        start = time.perf_counter()
        matrix_mul[blocks, threads_per_block](d_a, d_b, d_c, n)
        d_c.copy_to_host(c)
        elapsed = (time.perf_counter() - start) * 1000


        if n != 1:
            print(f"bench9 - Size: {n}x{n} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench9_{n}x{n}',
                'size': n,
                'gpu_time_ms': elapsed
            })

    return results

def bench10():
    # 1 is for precompilation
    sizes = [1, 16, 64, 256, 512, 1024, 2048, 4096]
    results = []

    for n in sizes:
        total = n * n

        a = np.array([(i * 12345.6789) % 1000 for i in range(total)], dtype=np.float32)
        b = np.array([(i * 98765.4321) % 1000 for i in range(total)], dtype=np.float32)
        c = np.zeros(total, dtype=np.float32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        threads_per_block = 256
        blocks = (total + threads_per_block - 1) // threads_per_block

        start = time.perf_counter()
        matrix_mul[blocks, threads_per_block](d_a, d_b, d_c, n)
        d_c.copy_to_host(c)
        elapsed = (time.perf_counter() - start) * 1000


        if n != 1:
            print(f"bench10 - Size: {n}x{n} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench10_{n}x{n}',
                'size': n,
                'gpu_time_ms': elapsed
            })

    return results