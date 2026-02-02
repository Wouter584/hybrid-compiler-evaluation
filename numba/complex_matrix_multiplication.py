import time
import numpy as np
from numba import cuda
from fast_fourier_transform import matrix_mul

def bench7():
    # 1 is for precompilation
    sizes = [1, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    for n in sizes:
        total = n * n

        a = np.array([complex(i * 12345.6789, 0.0) for i in range(total)], dtype=complex)
        b = np.array([complex(0.0, i * 98765.4321)  for i in range(total)], dtype=complex)
        c = np.zeros(total, dtype=complex)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        threads_per_block = 256
        blocks = (total + threads_per_block - 1) // threads_per_block

        start = time.perf_counter()
        matrix_mul[blocks, threads_per_block](d_c, d_a, d_b, n, n)
        d_c.copy_to_host(c)
        elapsed = (time.perf_counter() - start) * 1000


        if n != 1:
            print(f"bench7 - Size: {n}x{n} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench7_{n}x{n}',
                'size': n,
                'gpu_time_ms': elapsed
            })

    return results

def bench8():
    n = 512
    total = n * n
    iters = [1, 1, 10, 20, 50, 100, 200]
    results = []
    first = True

    a = np.array([complex(i * 12345.6789, 0.0) for i in range(total)], dtype=complex)
    b = np.array([complex(0.0, i * 98765.4321)  for i in range(total)], dtype=complex)
    c = np.zeros(total, dtype=complex)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(c)

    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block

    for iter in iters:
        start = time.perf_counter()
        for _ in range(iter):
            matrix_mul[blocks, threads_per_block](d_c, d_a, d_b, n, n)
            cuda.synchronize()

        d_c.copy_to_host(c)

        elapsed = (time.perf_counter() - start) * 1000
        if not first:
            print(f"bench8 - Size: {n}x{n}_{iter} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench8_{n}x{n}_{iter}',
                'size': iter,
                'gpu_time_ms': elapsed
            })
        first = False
    return results

if __name__ == '__main__':
    bench7()
    bench8()