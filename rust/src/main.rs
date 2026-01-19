#![engine(cuda::engine)]

mod mandelbrot;
mod fast_fourier_transform;
use cuda::dmem::{Buffer, DSend};
use cuda::{device_sync, gpu};
use std::fs::File;
use std::io::prelude::*;


#[derive(Debug, Clone)]
pub struct BenchResults {
    pub gpu_time: f64,
    pub test: String,
    pub size: usize,
}

use std::time::Instant;

#[kernel]
unsafe fn matrix_mul(a: &[i32], b: &[i32], mut c: Buffer<i32>, n: i32) {
    let tid = gpu::global_tid_x();
    let i = (tid / n) as usize;
    let j = (tid % n) as usize;
    let n_usize = n as usize;
    if i < n_usize && j < n_usize {
        let mut sum = 0;
        for k in 0..n_usize {
            sum += a[i * n_usize + k] * b[k * n_usize + j];
        }
        c.set(i * n_usize + j, sum);
    }
}

pub fn bench1() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();
    let sizes = [16, 64, 256, 512, 1024, 2048, 4096, 8192];

    for &n in &sizes {
        let total = (n * n) as usize;

        // Generate deterministic pseudo-random matrix data
        let mut a = vec![0; n * n];
        let mut b = vec![0; n * n];
        for i in 0..n * n {
            // random values
            let v1 = (i * 1234567) % 1000;
            let v2 = (i * 7654321) % 1000;
            a[i] = v1 as i32;
            b[i] = v2 as i32;
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let buf_c = Buffer::<i32>::alloc(total).unwrap();
        let mut d_buf_c = buf_c.to_device().unwrap();

        let threads_per_block = 64;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = Instant::now();
        let mut d_n = (n as i32).to_device().unwrap();
        match matrix_mul.launch_with_dptr(
            threads_per_block,
            blocks,
            &mut d_a,
            &mut d_b,
            &mut d_buf_c,
            &mut d_n
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error launching matrix mul kernel: {:?}", e);
            }
        }

        let x = buf_c.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("{}x{}", n, n),
            size: n,
        });
    }

    println!("Bench 1 Results:");
    for result in &results {
        println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    }

    results
}

pub fn bench2() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();

    let n = 1024;
    let total = n * n;
    let iters = [1, 10, 20, 50, 100, 200, 500];

    // Generate deterministic pseudo-random matrix data
    let mut a = vec![0; n * n];
    let mut b = vec![0; n * n];
    for i in 0..n * n {
        // random values
        let v1 = (i * 1234567) % 1000;
        let v2 = (i * 7654321) % 1000;
        a[i] = v1 as i32;
        b[i] = v2 as i32;
    }

    let mut d_a = a.as_slice().to_device().unwrap();
    let mut d_b = b.as_slice().to_device().unwrap();
    let mut d_buf_c = Buffer::<i32>::alloc(total).unwrap().to_device().unwrap();

    let threads_per_block = 64;
    let blocks = (total + threads_per_block - 1) / threads_per_block;

    for &iter in &iters {
        let start = Instant::now();
        let mut d_n = (n as i32).to_device().unwrap();
        for _ in 0..iter {
            match matrix_mul.launch_with_dptr(
                threads_per_block,
                blocks,
                &mut d_a,
                &mut d_b,
                &mut d_buf_c,
                &mut d_n
            ) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error launching matrix mul kernel: {:?}", e);
                }
            }
            match device_sync() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error synchronizing device after fft kernel: {:?}", e);
                }
            }
        }
        d_buf_c.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("{}x{}", n, n),
            size: iter,
        });
    }

    println!("\nBench 2 Results:");
    for result in &results {
        println!("Test: {} | {} iterations | GPU Time: {:.3} ms", result.test, result.size, result.gpu_time);
    }

    results
}



fn main() {
    // Run the benchmarks
    println!("Running Rust benchmarks...");
    println!("Running Bench 1...");
    let results1 = bench1();
    println!("Running Bench 2...");
    let results2 = bench2();
    println!("Running Bench 3...");
    let results3 = mandelbrot::bench3();
    println!("Running Bench 4...");
    let results4 = mandelbrot::bench4();
    println!("Running Bench 5...");
    let results5 = fast_fourier_transform::bench5();

    // write the results to a file
    // in a format that can be read by python
    // such as json
    let mut res = "".to_string();

    for result in &results1 {
        res.push_str(&format!("B1|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }
    for result in &results2 {
        res.push_str(&format!("B2|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }
    for result in &results3 {
        res.push_str(&format!("B3|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }
    for result in &results4 {
        res.push_str(&format!("B4|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }
    for result in &results5 {
        res.push_str(&format!("B5|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }

    let mut file = File::create("results/bench_results.txt").unwrap();
    file.write_all(res.as_bytes()).unwrap();
    println!("Benchmarks complete. Results written to bench_results.txt");
}
