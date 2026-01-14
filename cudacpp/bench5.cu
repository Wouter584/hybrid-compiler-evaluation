#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <cufft.h>

struct C64 {
    double real;
    double imaginary;
};

__host__ __device__
C64 operator+(const C64& a, const C64& b)
{
    return C64{a.real + b.real, a.imaginary + b.imaginary};
}

__host__ __device__
C64 operator-(const C64& a, const C64& b)
{
    return C64{a.real - b.real, a.imaginary - b.imaginary};
}

__host__ __device__
C64 operator*(const C64& a, const C64& b)
{
    return C64{a.real * b.real - a.imaginary * b.imaginary,
               a.real * b.imaginary + a.imaginary * b.real};
}

__host__ __device__
C64 c64_exp(const C64& a)
{
    const double exp_x = exp(a.real);
    return C64{exp_x * cos(a.imaginary), exp_x * sin(a.imaginary)};
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matrix_mul(C64* result, const C64* a, const C64* b, const int width, const int height) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / height;
    const int j = t % height;
    auto sum = C64{0.0, 0.0};
    for (int k = 0; k < width; ++k) {
        sum = sum + a[i * width + k] * b[k * height + j];
    }
    result[i * height + j] = sum;
}

__global__ void fft_kernel(C64* result, const C64* x_input, const int width, const int height) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / (height >> 1);
    const int j = t % (height >> 1);
    const auto temp = c64_exp(C64{0.0, -M_PI * i / width})
        * x_input[i * height + j + (height >> 1)];
    result[t] = x_input[i * height + j] + temp;
    result[(i + width) * (height >> 1) + j] = x_input[i * height + j] - temp;
}

struct BenchResult {
    int size{};
    double gpu_time{};
    std::string title;
};

struct FourierResult
{
    std::vector<BenchResult> bench_result;
    std::vector<C64> transform_result;
};

FourierResult fast_fourier_transform(const std::vector<C64> &x, int width)
{
    const int n = x.size();

    if (std::__popcount(n) != 1)
        throw std::invalid_argument("size of x must be a power of 2");

    int height = n / width;

    std::vector<BenchResult> bench_results(log2(height) + 1);
    std::vector<C64> result(n), new_x(n), matrix(width * width);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j)
        {
            matrix[i * width + j] = c64_exp(C64{0.0, -2.0 * M_PI * i * j / width});
        }
    }

    const int x_size = n * sizeof(C64);
    const int matrix_size = width * width * sizeof(C64);

    C64 *d_new_x, *d_matrix, *d_x;
    CHECK_CUDA(cudaMalloc(&d_new_x, x_size));
    CHECK_CUDA(cudaMalloc(&d_matrix, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_x, x_size));

    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(), matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x_size, cudaMemcpyHostToDevice));

    constexpr int threads_per_block = 256;
    int blocks = n / threads_per_block;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    matrix_mul<<<blocks, threads_per_block>>>(d_new_x, d_matrix, d_x, width, height);
    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

    bench_results[0].size = 0;
    bench_results[0].gpu_time = gpu_time;
    bench_results[0].title = std::to_string(width) + "x" + std::to_string(height);

    CHECK_CUDA(cudaFree(d_matrix));

    C64* d_buffer_1 = d_x;
    C64* d_buffer_2 = d_new_x;

    // Bit shift one to the right == division by 2.
    blocks = blocks >> 1;
    int counter = 1;

    while (width < n) {
        gpu_start = std::chrono::high_resolution_clock::now();

        fft_kernel<<<blocks, threads_per_block>>>(d_buffer_1, d_buffer_2, width, height);
        cudaDeviceSynchronize();

        gpu_end = std::chrono::high_resolution_clock::now();
        gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

        bench_results[counter].size = counter;
        bench_results[counter].gpu_time = gpu_time;
        bench_results[counter].title = std::to_string(width) + "x" + std::to_string(height);

        // Switch the buffers, new_x becomes x (input for the next iteration),
        // x becomes new_x (values will be overwritten).
        std::swap(d_buffer_1, d_buffer_2);
        width <<= 1;
        height >>= 1;
        counter += 1;
    }
    CHECK_CUDA(cudaMemcpy(result.data(), d_buffer_2, x_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_buffer_1));
    CHECK_CUDA(cudaFree(d_buffer_2));

    return FourierResult{bench_results, result};
}

void export_bench_results(const BenchResult* results, const int count) {
    // open a file to write the results__host__
    std::ofstream file("results/bench_results5.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing results." << std::endl;
        return;
    }
    for (int i = 0; i < count; ++i) {
        file << "B5|" << results[i].title << "|"
             << results[i].size << "|"
             << results[i].gpu_time << std::endl;
    }
    file.close();
    std::cout << "Results exported to bench_results5.txt" << std::endl;
}

int main() {
    constexpr int n = 8192*16*16;

    // Create random inputs
    std::vector<cufftDoubleComplex> complex_pseudo_random_input(n);
    constexpr size_t complex_size = n * sizeof(cufftDoubleComplex);
    std::vector<C64> pseudo_random_input(n);
    for (int i = 0; i < n; ++i) {
        const double input_value = i * 12345.6789;

        complex_pseudo_random_input[i].x = input_value;
        complex_pseudo_random_input[i].y = 0.0;
        pseudo_random_input[i] = C64{input_value, 0.0};
    }

    // Calculate with cufft
    cufftDoubleComplex *d_in, *d_out;
    cudaMalloc(&d_in, complex_size);
    cudaMalloc(&d_out, complex_size);
    cudaMemcpy(d_in, complex_pseudo_random_input.data(), complex_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    cudaMemcpy(complex_pseudo_random_input.data(), d_out, complex_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);

    // Calculate with fast_fourier_transform method
    constexpr int width = 32;
    const auto output = fast_fourier_transform(pseudo_random_input, width);

    double max_distance = 0.0;
    for (size_t i = 0; i < output.transform_result.size() && i < complex_pseudo_random_input.size(); ++i) {
        const double dx = output.transform_result[i].real - complex_pseudo_random_input[i].x;
        const double dy = output.transform_result[i].imaginary - complex_pseudo_random_input[i].y;
        const double distance = std::sqrt(dx * dx + dy * dy);

        if (distance > max_distance) {
            max_distance = distance;
        }
    }
    std::cout << "Maximum distance from cufft: (" << max_distance << ")\n";

    export_bench_results(output.bench_result.data(), log2(n/width) + 1);
    return 0;
}

