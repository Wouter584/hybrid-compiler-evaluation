# This python file is used to run the benchmarks
# of Rust-gpuhc, CUDA C++, and Numba Python.
# it publishes a complete report of the benchmarks
# including figures

import os
import subprocess
import sys
import time
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
        "B7": [],
        "B8": [],
        "B9": [],
        "B10": [],
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
    subprocess.run(["nvcc", "-o", "b6", "bench6.cu"], check=True)
    subprocess.run(["./b6"])
    subprocess.run(["nvcc", "-o", "b7", "bench7.cu"], check=True)
    subprocess.run(["./b7"])
    subprocess.run(["nvcc", "-o", "b8", "bench8.cu"], check=True)
    subprocess.run(["./b8"])
    subprocess.run(["nvcc", "-o", "b9", "bench9.cu"], check=True)
    subprocess.run(["./b9"])
    subprocess.run(["nvcc", "-o", "b10", "bench10.cu"], check=True)
    subprocess.run(["./b10"])
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
        "B6": [],
        "B7": [],
        "B8": [],
        "B9": [],
        "B10": [],
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
    with open("cudacpp/results/bench_results6.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results7.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results8.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results9.txt", "r") as f:
        results += "\n" + f.read()
    with open("cudacpp/results/bench_results10.txt", "r") as f:
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
        "B6": [],
        "B7": [],
        "B8": [],
        "B9": [],
        "B10": [],
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

def run_all_benchmarks(iterations:int):
    result_dict = {
        "rust": [],
        "cuda": [],
        "numba": [],
    } # Structure: {language: [{bench_name: [(size, average_time)]}]}
    start = time.perf_counter()
    for iteration in range(iterations):
        run_rust_benchmarks()
        result_dict["rust"].append(get_rust_results())
        run_cuda_benchmarks()
        result_dict["cuda"].append(get_cuda_results())
        run_numba_benchmarks()
        result_dict["numba"].append(get_numba_results())
        print("Finished benchmarks iteration:", iteration)
    elapsed = (time.perf_counter() - start) / 60
    print("Elapsed minutes for all benchmarks:", elapsed)

    ordered_results = {
        "rust": {},
        "cuda": {},
        "numba": {},
    } # Structure: {language: {bench_name: {size, [gpu_time]}}}
    for language, results in result_dict.items():
        for bench_dict in results:
            for bench_name, bench_results in bench_dict.items():
                if bench_name not in ordered_results[language]:
                    ordered_results[language][bench_name] = {}
                for size, gpu_time in bench_results:
                    if size not in ordered_results[language][bench_name]:
                        ordered_results[language][bench_name][size] = [gpu_time]
                        continue
                    else:
                        ordered_results[language][bench_name][size].append(gpu_time)

    results_to_file(ordered_results, "all_results.txt")


def results_to_file(results, file_name):
    """
    Output results as a file.
    :param results: Structure: {language: {bench_name: {size, [gpu_time]}}}
    :param file_name:
    :return:
    """
    with open(file_name, "w") as f:
        for language, language_results_dict in results.items():
            for bench_name, bench_results_dict in language_results_dict.items():
                for size, gpu_times in bench_results_dict.items():
                    f.write(f"{language}|{bench_name}|{size}|")
                    for gpu_time in gpu_times:
                        f.write(f"{gpu_time},")
                    f.write(f"\n")

def file_to_results(file_name):
    """
    Returns results from a file.
    :param file_name:
    :return: Structure: {language: {bench_name: {size, [gpu_time]}}}
    """
    results = {}
    with open(file_name, "r") as f:
        contents = f.read()
    lines = contents.split("\n")
    for line in lines:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the language
        language = line[0]
        # second element is the benchmark name
        bench_name = line[1]
        # third element is the size
        bench_size = line[2]
        # fourth element is a list of times separated by a comma, without the last empty element.
        bench_times = line[3].split(",")[:-1]
        if language not in results:
            results[language] = {}
        if bench_name not in results[language]:
            results[language][bench_name] = {}
        if bench_size not in results[language][bench_name]:
            results[language][bench_name][bench_size] = []

        for time in bench_times:
            results[language][bench_name][bench_size].append(float(time))

    return results

def results_as_tuples(results):
    """
    Convert results to a single value for each bench (tuple instead of dict).
    :param results: Structure: {language: {bench_name: {size, [gpu_time]}}}
    :return: Structure: {language: {bench_name: [(size, average_time)]}}
    """
    results_as_tuple = {}
    for language, language_results_dict in results.items():
        for bench_name, bench_results_dict in language_results_dict.items():
            for bench_size, bench_times in bench_results_dict.items():
                if language not in results_as_tuple:
                    results_as_tuple[language] = {}
                if bench_name not in results_as_tuple[language]:
                    results_as_tuple[language][bench_name] = []
                results_as_tuple[language][bench_name].append((bench_size, bench_times[0]))
    return results_as_tuple


def get_all_averages():
    average_results = {} # Structure: {language: {bench_name: {size, [gpu_time]}}}
    all_results = file_to_results("all_results.txt")

    for language, language_results_dict in all_results.items():
        for bench_name, bench_results_dict in language_results_dict.items():
            for bench_size, bench_times in bench_results_dict.items():
                if language not in average_results:
                    average_results[language] = {}
                if bench_name not in average_results[language]:
                    average_results[language][bench_name] = {}
                if bench_size not in average_results[language][bench_name]:
                    average_results[language][bench_name][bench_size] = []

                average_time = 0.0
                for time in bench_times:
                    average_time += time
                average_time = average_time / len(bench_times)
                average_results[language][bench_name][bench_size].append(average_time)

    results_to_file(average_results, "averaged_results")
    return results_as_tuples(average_results)

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
    run_all_benchmarks(10)
    averages = get_all_averages()
    results_rust = averages["rust"]
    results_cuda = averages["cuda"]
    results_numba = averages["numba"]
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
    plot_benchmarks_baseline(results_cuda["B6"], {
        "CUDA": results_cuda["B6"],
        "Numba": results_numba["B6"],
    }, file_name="B6", title="Sum reduction", xlabel="Number of iterations")
    plot_benchmarks_baseline(results_cuda["B7"], {
        "Rust": results_rust["B7"],
        "CUDA": results_cuda["B7"],
        "Numba": results_numba["B7"],
    }, file_name="B7", title="Complex matrix multiplication", xlabel="Matrix size")
    plot_benchmarks_baseline(results_cuda["B8"], {
        "Rust": results_rust["B8"],
        "CUDA": results_cuda["B8"],
        "Numba": results_numba["B8"],
    }, file_name="B8", title="Complex matrix multiplication", xlabel="Number of iterations")
    plot_benchmarks_baseline(results_cuda["B9"], {
        "Rust": results_rust["B9"],
        "CUDA": results_cuda["B9"],
        "Numba": results_numba["B9"],
    }, file_name="B9", title="Double precision float matrix multiplication", xlabel="Matrix size")
    plot_benchmarks_baseline(results_cuda["B10"], {
        "Rust": results_rust["B10"],
        "CUDA": results_cuda["B10"],
        "Numba": results_numba["B10"],
    }, file_name="B10", title="Single precision float matrix multiplication", xlabel="Matrix size")