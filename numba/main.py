import numpy as np
from numba import cuda
import time
import mandelbrot

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

def bench1():
    # 1 is for precompilation
    sizes = [1, 16, 64, 256, 512, 1024, 2048, 4096, 8192]
    results = []

    for n in sizes:
        total = n * n

        a = np.array([(i * 1234567) % 1000 for i in range(total)], dtype=np.int32)
        b = np.array([(i * 7654321) % 1000 for i in range(total)], dtype=np.int32)
        c = np.zeros(total, dtype=np.int32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        threads_per_block = 64
        blocks = (total + threads_per_block - 1) // threads_per_block

        start = time.perf_counter()
        matrix_mul[blocks, threads_per_block](d_a, d_b, d_c, n)
        d_c.copy_to_host(c)
        elapsed = (time.perf_counter() - start) * 1000


        if n != 1:
            print(f"bench1 - Size: {n}x{n} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench1_{n}x{n}',
                'size': n,
                'gpu_time_ms': elapsed
            })

    return results

def bench2():
    n = 1024
    total = n * n
    iters = [1, 1, 10, 20, 50, 100, 200, 500]
    results = []
    first = True
    for iter in iters:
        a = np.arange(total, dtype=np.int32)
        b = np.arange(total, dtype=np.int32)
        c = np.zeros(total, dtype=np.int32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        threads_per_block = 64
        blocks = (total + threads_per_block - 1) // threads_per_block

        start = time.perf_counter()
        for _ in range(iter):
            matrix_mul[blocks, threads_per_block](d_a, d_b, d_c, n)
            cuda.synchronize()

        d_c.copy_to_host(c)

        elapsed = (time.perf_counter() - start) * 1000
        if not first:
            print(f"bench2 - Size: {n}x{n}_{iter} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench2_{n}x{n}_{iter}',
                'size': iter,
                'gpu_time_ms': elapsed
            })
        first = False
    return results

if __name__ == "__main__":
    res1 = bench1()
    res2 = bench2()
    res3 = mandelbrot.bench3()
    res4 = mandelbrot.bench4()

    # Save results to a file
    with open("results/bench_results.txt", "w") as f:
        for res in res1:
            f.write(f"B1|{res['test']}|{res['size']}|{res['gpu_time_ms']}\n")
        for res in res2:
            f.write(f"B2|{res['test']}|{res['size']}|{res['gpu_time_ms']}\n")
        for res in res3:
            f.write(f"B3|{res['test']}|{res['max_iterations']}|{res['gpu_time_ms']}\n")
        for res in res4:
            f.write(f"B4|{res['test']}|{res['iterations']}|{res['gpu_time_ms']}\n")
