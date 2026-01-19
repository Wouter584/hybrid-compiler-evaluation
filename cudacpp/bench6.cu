#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <numeric>
using namespace std;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct BenchResult {
    int size;
    double gpu_time;
    std::string title;
};

void export_bench_results(const BenchResult* results, const int count) {
    // open a file to write the results
    std::ofstream file("results/bench_results6.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing results." << std::endl;
        return;
    }
    // write as follows: B1|<bench_name>|<size>|<gpu_time>
    for (int i = 0; i < count; ++i) {
        file << "B6|" << results[i].title << "|"
             << results[i].size << "|"
             << results[i].gpu_time << std::endl;
    }
    file.close();
    std::cout << "Results exported to bench_results6.txt" << std::endl;
}

// Simple reduction adapted from https://blog.damavis.com/en/cuda-tutorial-warp/
__global__ void sum_kernel(const float* input_array, float* out)
{
    const int idx = threadIdx.x;
    float val = input_array[idx];
    __shared__ float shared_mem[8];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(idx%32==0) {
        shared_mem[idx>>5]=val;
    }
    __syncthreads();
    if (idx<=8) {
        val=shared_mem[idx];
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (idx==0) {
            *out = val;
        }
    }
}


int main() {
    constexpr int iterations_list[] = {1, 10, 100, 1000, 10000, 100000, 1000000};
    constexpr int iterations_list_count = sizeof(iterations_list) / sizeof(iterations_list[0]);
    BenchResult bench_results[iterations_list_count];

    constexpr int size = 256;
    std::vector<float> pseudo_random_input(size);
    float output;

    for (int i = 0; i < size; ++i) {
        pseudo_random_input[i] = i * 123.0f;
    }
    const float expected = accumulate(begin(pseudo_random_input), end(pseudo_random_input),0, plus<float>());

    float *d_pseudo_random_input, *d_output;
    constexpr int input_size = size * sizeof(float);
    constexpr int output_size = sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_pseudo_random_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));
    CHECK_CUDA(cudaMemcpy(d_pseudo_random_input, pseudo_random_input.data(), input_size, cudaMemcpyHostToDevice));

    for (int s = 0; s < iterations_list_count; ++s) {
        const int iterations = iterations_list[s];

        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            sum_kernel<<<1, size>>>(d_pseudo_random_input, d_output);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        CHECK_CUDA(cudaMemcpy(&output, d_output, output_size, cudaMemcpyDeviceToHost));
        auto gpu_end = std::chrono::high_resolution_clock::now();
        const auto gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        bench_results[s].size = iterations;
        bench_results[s].gpu_time = gpu_time;
        bench_results[s].title = "parallel_sum_result:" + std::to_string(output);

        std::cout << "Sum result: " << std::to_string(output) <<
                     ", expected: " << std::to_string(expected) << std::endl;
    }

    CHECK_CUDA(cudaFree(d_pseudo_random_input));
    CHECK_CUDA(cudaFree(d_output));

    export_bench_results(bench_results, iterations_list_count);
    return 0;
}

