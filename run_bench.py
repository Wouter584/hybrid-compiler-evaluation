# This python file is used to run the benchmarks
# of Rust-gpuhc, CUDA C++, and Numba Python.
# it publishes a complete report of the benchmarks
# including figures

import os
import subprocess
import sys

import matplotlib.pyplot as plt

import re
from matplotlib.ticker import StrMethodFormatter, NullFormatter

def run_rust_benchmarks():
    """
    Run the Rust benchmarks
    """
    print("Running Rust benchmarks...")
    os.chdir("rust")
    subprocess.run(["cargo", "run", "--release"], check=True)
    os.chdir("..")

def get_rust_results():

    # retrieve the results
    # they are stored in bench_results.txt
    rust_results = {
        "B1": [],
        "B2": [],
        "B3": [],
        "B4": [],
        "B5": [],
    }
    results = ""
    with open("rust/results/bench_results.txt", "r") as f:
        results = f.read()

    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        rust_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return rust_results

def run_cuda_benchmarks():
    """
    Run the CUDA benchmarks
    """
    print("Running CUDA benchmarks...")
    os.chdir("cudacpp")
    subprocess.run(["nvcc", "-o", "b1", "bench1.cu"], check=True)
    subprocess.run(["./b1"])
    subprocess.run(["nvcc", "-o", "b2", "bench2.cu"], check=True)
    subprocess.run(["./b2"])
    subprocess.run(["nvcc", "-o", "b3", "bench3.cu"], check=True)
    subprocess.run(["./b3"])
    subprocess.run(["nvcc", "-o", "b4", "bench4.cu"], check=True)
    subprocess.run(["./b4"])
    subprocess.run(["nvcc", "-o", "b5", "bench5.cu", "-lcufft"], check=True)
    subprocess.run(["./b5"])
    os.chdir("..")

def get_cuda_results():
    # retrieve the results
    # they are stored in bench_results.txt
    cuda_results = {
        "B1": [],
        "B2": [],
        "B3": [],
        "B4": [],
        "B5": [],
    }
    results = ""
    with open("cudacpp/results/bench_results1.txt", "r") as f:
        results = f.read()
    with open("cudacpp/results/bench_results2.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results3.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results4.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results5.txt", "r") as f:
        results += "\n" + f.read()
    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        cuda_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return cuda_results

def run_numba_benchmarks():
    """
    Run the Numba benchmarks
    """
    print("Running Numba benchmarks...")
    os.chdir("numba")
    # run the benchmarks
    subprocess.run([sys.executable, "main.py"], check=True)
    os.chdir("..")


def get_numba_results():
    # retrieve the results
    # they are stored in bench_results.txt
    numba_results = {
        "B1": [],
        "B2": [],
        "B3": [],
        "B4": [],
        "B5": [],
    }
    results = ""
    with open("numba/results/bench_results.txt", "r") as f:
        results = f.read()
    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        numba_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return numba_results


def plot_benchmarks(all_benchmarks, file_name="combined_benchmarks.png", title="Benchmark Results",
                    xlabel="Size", ylabel="Time (ms)"):
    plt.figure(figsize=(10, 6))

    for lang, benchmark_dict in all_benchmarks.items():
        sizes = []
        times = []

        for (size, time) in benchmark_dict:
            sizes.append(size)
            times.append(time)


        plt.plot(sizes, times, marker='o', label=lang)

    # ax = plt.gca()
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # ax.yaxis.set_minor_formatter(NullFormatter())

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    print("Close the plot to continue...")
    plt.show()


def plot_benchmarks_baseline(baseline, all_benchmarks, file_name="combined_benchmarks.png", title="Benchmark Results",
                             xlabel="Size", ylabel="Relative time (lower is better)"):
    plt.figure(figsize=(10, 6))

    baseline_times = {}
    for benchmark in baseline:
        # benchmark is a tuple of (size, time)
        size, time = benchmark
        baseline_times[size] = time

    for lang, benchmark_dict in all_benchmarks.items():
        sizes = []
        times = []

        for (size, time) in benchmark_dict:
            sizes.append(size)
            times.append(time / baseline_times[size])


        plt.plot(sizes, times, marker='o', label=lang)

    # ax = plt.gca()
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # ax.yaxis.set_minor_formatter(NullFormatter())

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    print("Close the plot to continue...")
    plt.show()

if __name__ == "__main__":
    run_cuda_benchmarks()
    run_numba_benchmarks()
    run_rust_benchmarks()
    results_rust = get_rust_results()
    results_cuda = get_cuda_results()
    results_numba = get_numba_results()
    # plot_benchmarks({
    #     "Rust": results_rust["B1"],
    #     "CUDA": results_cuda["B1"],
    #     "Numba": results_numba["B1"],
    # }, title="Matrix multiplication", xlabel="Matrix size")
    plot_benchmarks_baseline(results_cuda["B1"], {
        "Rust": results_rust["B1"],
        "CUDA": results_cuda["B1"],
        "Numba": results_numba["B1"],
    }, file_name="B1", title="Matrix multiplication", xlabel="Matrix size")
    plot_benchmarks_baseline(results_cuda["B2"], {
        "Rust": results_rust["B2"],
        "CUDA": results_cuda["B2"],
        "Numba": results_numba["B2"],
    }, file_name="B2", title="Matrix multiplication", xlabel="Number of iterations")
    plot_benchmarks_baseline(results_cuda["B3"], {
        "Rust": results_rust["B3"],
        "CUDA": results_cuda["B3"],
        "Numba": results_numba["B3"],
    }, file_name="B3", title="Mandelbrot set generation", xlabel="Maximum number of iterations")
    plot_benchmarks_baseline(results_cuda["B4"], {
        "Rust": results_rust["B4"],
        "CUDA": results_cuda["B4"],
        "Numba": results_numba["B4"],
    }, file_name="B4", title="Mandelbrot set generation", xlabel="Number of iterations")
    plot_benchmarks_baseline(results_cuda["B5"], {
        "Rust": results_rust["B5"],
        "CUDA": results_cuda["B5"],
        "Numba": results_numba["B5"],
    }, file_name="B5", title="Fast fourier transform", xlabel="Iteration")