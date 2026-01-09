#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

// For generating the png (available at https://github.com/nothings/stb/blob/master/stb_image_write.h)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

struct Complex {
    double real;
    double imaginary;
};

__device__ bool has_escaped(const Complex& c) {
    return c.real * c.real + c.imaginary * c.imaginary >= 4.0;
}

__device__ Complex iterate(const Complex& z, const Complex& c)
{
    const double real = (z.real * z.real) - (z.imaginary * z.imaginary) + c.real;
    const double imaginary = (z.real + z.real) * z.imaginary + c.imaginary;
    return Complex{real, imaginary};
}

__device__ uint8_t mandel(const Complex& c, const int& max_iters)
{
    Complex z = c;
    for (int i = 0; i < max_iters; ++i) {
        if (has_escaped(z))
        {
            const uint8_t color = (255 * i) / max_iters;
            return color;
        }
        z = iterate(z, c);

    }
    return 255;
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void create_mandelbrot_fractal(uint8_t* image, const int& pixel_span, const double& coordinate_span,
    const double& real_start, const double& i_start, const int& max_iterations) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_pixel = tid / pixel_span;
    const int y_pixel = tid % pixel_span;
    const double step = coordinate_span / pixel_span;

    if (x_pixel < pixel_span && y_pixel < pixel_span) {
        const double x = real_start + (x_pixel * step);
        const double y = i_start + (y_pixel * step);
        const auto c = Complex{x, y};
        image[y_pixel * pixel_span + x_pixel] = mandel(c, max_iterations);
    }
}

struct BenchResult {
    int size{};
    double gpu_time{};
    std::string title;
};

void export_bench_results(const BenchResult* results, const int count) {
    // open a file to write the results
    std::ofstream file("results/bench_results4.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing results." << std::endl;
        return;
    }
    // write as follows: B4|<bench_name>|<size>|<gpu_time>
    for (int i = 0; i < count; ++i) {
        file << "B4|" << results[i].title << "|"
             << results[i].size << "|"
             << results[i].gpu_time << std::endl;
    }
    file.close();
    std::cout << "Results exported to bench_results4.txt" << std::endl;
}

int main() {
    constexpr int iterations_list[] = {10, 20, 50, 100};
    constexpr int iterations_list_count = sizeof(iterations_list) / sizeof(iterations_list[0]);
    BenchResult bench_results[iterations_list_count];
    const int max_iterations = 64;

    constexpr int pixel_span = 8192;
    std::vector<uint8_t> image(pixel_span*pixel_span);
    constexpr size_t size = pixel_span * pixel_span * sizeof(uint8_t);

    constexpr int threads_per_block = 256;
    constexpr int blocks = (pixel_span*pixel_span + threads_per_block - 1) / threads_per_block;

    for (int s = 0; s < iterations_list_count; ++s) {
        uint8_t *d_image;
        CHECK_CUDA(cudaMalloc(&d_image, size));

        auto gpu_start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations_list[s]; ++t)
        {
            create_mandelbrot_fractal<<<blocks, threads_per_block>>>(d_image, pixel_span,
                3.0, -2.0, -1.5, max_iterations);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(image.data(), d_image, size, cudaMemcpyDeviceToHost));
        }
        auto gpu_end = std::chrono::high_resolution_clock::now();
        const auto gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "Max iterations: " << max_iterations << ", GPU time: " << gpu_time << " ms" << std::endl;

        CHECK_CUDA(cudaFree(d_image));

        bench_results[s].size = iterations_list[s];
        bench_results[s].gpu_time = gpu_time;
        bench_results[s].title = std::to_string(pixel_span) + "x" + std::to_string(pixel_span) + "_max_iterations_" + std::to_string(max_iterations);
    }

    stbi_write_png("./results/mandelbrot.png", pixel_span, pixel_span, 1, image.data(), pixel_span);

    export_bench_results(bench_results, iterations_list_count);
    return 0;
}

