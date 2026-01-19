import time
import numpy as np
from numba import cuda

@cuda.jit
def sum_kernel(input_array, out):
    idx = cuda.threadIdx.x
    val = input_array[idx]
    shared = cuda.shared.array(8, dtype=np.float32)
    offset = 16
    while offset > 0:
        val += cuda.shfl_down_sync(0xffffffff, val, offset)
        offset //= 2
    if idx % 32 == 0:
        shared[idx >> 5] = val
    cuda.syncthreads()
    if idx <= 8:
        val = shared[idx]
        val += cuda.shfl_down_sync(0xffffffff, val, 4)
        val += cuda.shfl_down_sync(0xffffffff, val, 2)
        val += cuda.shfl_down_sync(0xffffffff, val, 1)
        if idx == 0:
            out[0] = val

@cuda.reduce
def sum_reduce(a, b):
    return a + b

def bench6():
    iterations_list = [1, 10, 100, 1000, 10000, 100000, 1000000]
    results = []
    size = 256
    pseudo_random_input = np.array([i * 123.0 for i in range(size)], dtype=np.float32)
    output = np.zeros(1, dtype=np.float32)
    expected = pseudo_random_input.sum()      # NumPy sum reduction

    # Precompile the kernels.
    d_temp1 = cuda.to_device(np.zeros(1, dtype=np.float32))
    d_temp2 = cuda.to_device(np.zeros(1, dtype=np.float32))
    sum_kernel[1, 1](d_temp1, d_temp2)
    cuda.synchronize()

    d_pseudo_random_input = cuda.to_device(pseudo_random_input)
    d_output = cuda.device_array_like(output)

    for iterations in iterations_list:
        # Reductions with Numba reduce:
        # start = time.perf_counter()
        # for _ in range(iterations):
        #     got = sum_reduce(pseudo_random_input)   # cuda sum reduction
        # elapsed = (time.perf_counter() - start) * 1000
        # print(f"bench6 - iterations: {iterations} | Time: {elapsed:.3f} ms")
        # print("Sum result:", got, ", expected:", expected)
        # results.append({
        #     'test': f'bench6_reduction',
        #     'size': iterations,
        #     'gpu_time_ms': elapsed
        # })
        start = time.perf_counter()
        for _ in range(iterations):
            sum_kernel[1, size](d_pseudo_random_input, d_output)
            cuda.synchronize()
        output = d_output.copy_to_host()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"bench6 - iterations: {iterations} | Time: {elapsed:.3f} ms")
        print("Sum result:", output[0], ", expected:", expected)
        results.append({
            'test': f'bench6_sum_result: {output[0]}',
            'iterations': iterations,
            'gpu_time_ms': elapsed
        })
    return results

if __name__ == '__main__':
    bench6()