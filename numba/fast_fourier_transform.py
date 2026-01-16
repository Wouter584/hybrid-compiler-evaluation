import cmath
import math
import time
import timeit
from xmlrpc.client import FastParser

from numba import cuda
import numpy as np

# FFT_vectorized from https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# fast_fourier_transform is a rewrite of FFT_vectorized, changed to use the GPU, without the use of numpy functions.
def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]

        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

@cuda.jit
def exp(x:np.complex128):
    exp_x = math.exp(x.real)
    return np.complex128(exp_x * math.cos(x.imag), exp_x * math.sin(x.imag))

@cuda.jit
def matrix_mul(result: np.ndarray, a: np.ndarray, b: np.ndarray, width: int, height: int):
    tid = cuda.grid(1)
    i = tid // height
    j = tid % height
    sum = 0
    for k in range(width):
        sum += a[i * width + k] * b[k * height + j]
    result[i * height + j] = sum

@cuda.jit
def fft_kernel(result: np.ndarray, x_input: np.ndarray, width: int, height: int):
    tid = cuda.grid(1)
    i = tid // (height >> 1)
    j = tid % (height >> 1)
    temp = (exp(-1j * math.pi * i / width)
            * x_input[i * height + j + (height >> 1)])
    result[tid] = x_input[i * height + j] + temp
    result[(i + width) * (height >> 1) + j] = x_input[i * height + j] - temp


def fast_fourier_transform(x, width: int):
    n = x.shape[0]
    results = []

    if math.log2(n) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # Precompile the kernels.
    d_temp1 = cuda.to_device(np.zeros(1, dtype=np.complex128))
    d_temp2 = cuda.to_device(np.zeros(1, dtype=np.complex128))
    d_temp3 = cuda.to_device(np.zeros(1, dtype=np.complex128))
    matrix_mul[1, 1](d_temp1, d_temp2, d_temp3, 0, 0)
    fft_kernel[1, 1](d_temp1, d_temp2, 0, 0)
    cuda.synchronize()

    height = n // width

    new_x = np.zeros(shape=n, dtype=np.complex128)
    matrix = np.zeros(shape=width*width, dtype=np.complex128)
    for i in range(width):
        for j in range(width):
            matrix[i * width + j] = cmath.exp(-2j * math.pi * i * j / width)

    d_new_x = cuda.device_array_like(new_x)
    d_matrix = cuda.to_device(matrix)
    d_x = cuda.to_device(x)

    threads_per_block = 256
    blocks = n // threads_per_block

    start = time.perf_counter()

    matrix_mul[blocks, threads_per_block](d_new_x, d_matrix, d_x, width, height)
    cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000

    print(f"bench5 - Matrix multiplication: {width}x{height} | Time: {elapsed:.3f} ms")
    results.append({
        'test': f'bench5_{width}x{height}',
        'iteration': 0,
        'gpu_time_ms': elapsed
    })

    d_buffer_1 = cuda.device_array_like(new_x)
    # Second buffer is the result from the matrix multiplication.
    d_buffer_2 = d_new_x

    # Bit shift one to the right == division by 2.
    blocks = blocks >> 1
    counter = 1
    
    while width < n:
        start = time.perf_counter()
        fft_kernel[blocks, threads_per_block](d_buffer_1, d_buffer_2, width, height)
        cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        print(f"bench5 - FFT kernel: {width}x{height} | Time: {elapsed:.3f} ms")
        results.append({
            'test': f'bench5_{width}x{height}',
            'iteration': counter,
            'gpu_time_ms': elapsed
        })

        # Switch the buffers, new_x becomes x (input for the next iteration),
        # x becomes new_x (values will be overwritten).
        d_buffer_1, d_buffer_2 = d_buffer_2, d_buffer_1
        width <<= 1
        height >>= 1
        counter += 1
    return d_buffer_2.copy_to_host(), results

def bench5():
    n = 8192*16*16
    pseudo_random_input = np.array([i * 12345.6789 for i in range(n)], dtype=np.complex128)
    (output, bench_result) = fast_fourier_transform(pseudo_random_input, 32)
    correct_result = np.fft.fft(pseudo_random_input)

    max_distance = 0
    for z1, z2 in zip(output, correct_result):
        dx = z1.real - z2.real
        dy = z1.imag - z2.imag
        distance = (dx*dx + dy*dy)**0.5
        if distance > max_distance:
            max_distance = distance
    print("Maximum distance from np.fft:", max_distance, "correct:", np.allclose(output, correct_result))
    return bench_result


if __name__ == '__main__':
    bench5()
    random_input = np.random.random(8192*16*16*16)

    print("FFT_vectorized correct: ", np.allclose(FFT_vectorized(random_input), np.fft.fft(random_input)))
    print("fast_fourier_transform correct:", np.allclose(fast_fourier_transform(random_input, 32)[0], np.fft.fft(random_input)))

    print("FFT_vectorized:", timeit.timeit("FFT_vectorized(random_input)",
                                         setup="from __main__ import FFT_vectorized; "
                                               "import numpy as np; random_input=np.random.random(8192*16*16*16*2)", number=1))
    print("fast_fourier_transform:", timeit.timeit("fast_fourier_transform(random_input, 32)",
                                         setup="from __main__ import fast_fourier_transform; "
                                               "import numpy as np; random_input=np.random.random(8192*16*16*16*2)", number=1))
    print("np.fft.fft:", timeit.timeit("np.fft.fft(random_input)",
                                         setup="import numpy as np; random_input=np.random.random(8192*16*16*16*2)", number=1))