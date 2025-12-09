"""
Compute and plot the Mandelbrot set using matplotlib.
Adapted from https://numba.pydata.org/numba-doc/0.17.0/user/examples.html
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from numba import cuda


@cuda.jit
def mandel(c: complex, max_iters):
    """
    Given a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    z = c
    for i in range(max_iters):
        if z.real * z.real + z.imag * z.imag >= 4:
            return (255 * i) // max_iters
        z = z*z + c

    return 255

@cuda.jit
def create_mandelbrot_fractal(image, pixel_span, coordinate_span, real_start, i_start, max_iterations):
    tid = cuda.grid(1)
    x_pixel = tid // pixel_span
    y_pixel = tid % pixel_span
    step = coordinate_span / pixel_span

    if x_pixel < pixel_span and y_pixel < pixel_span:
        x = real_start + x_pixel*step
        y = i_start + y_pixel * step
        image[y_pixel, x_pixel] = mandel(complex(x, y), max_iterations)

def bench3():
    # 1 is for precompilation
    max_iterations_list = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results = []

    pixel_span = 8192
    image = np.zeros((pixel_span, pixel_span), dtype=np.uint8)
    d_image = cuda.device_array_like(image)

    threads_per_block = 256
    blocks = (pixel_span*pixel_span + threads_per_block - 1) // threads_per_block

    for max_iterations in max_iterations_list:
        start = time.perf_counter()
        create_mandelbrot_fractal[blocks, threads_per_block](d_image, pixel_span, 3.0, -2.0, -1.5, max_iterations)
        cuda.synchronize()
        d_image.copy_to_host(image)
        elapsed = (time.perf_counter() - start) * 1000

        if max_iterations != 1:
            print(f"bench3 - Iteration: {max_iterations} | Time: {elapsed:.3f} ms")
            results.append({
                'test': f'bench3_{pixel_span}x{pixel_span}',
                'max_iterations': max_iterations,
                'gpu_time_ms': elapsed
            })

    plt.imsave("./results/mandelbrot.png",  image, cmap='gray')

    return results

if __name__ == '__main__':
    bench3()