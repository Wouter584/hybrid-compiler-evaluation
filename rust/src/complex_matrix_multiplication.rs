use cuda::dmem::{Buffer, DSend, DeepCopy};
use cuda::{device_sync, gpu};
use std::time::Instant;
use crate::BenchResults;
use crate::fast_fourier_transform::C64;
use crate::fast_fourier_transform::matrix_mul;

pub fn bench7() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();
    let sizes: [usize;7] = [16, 64, 256, 512, 1024, 2048, 4096];

    for n in sizes {
        let total = n * n;

        // Generate deterministic pseudo-random matrix data
        let mut a = vec![C64::new(0.0, 0.0); n * n];
        let mut b = vec![C64::new(0.0, 0.0); n * n];
        for i in 0..n * n {
            a[i] = C64::new(i as f64 * 12345.6789, 0.0);
            b[i] = C64::new(0.0, i as f64 * 98765.4321);
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let buf_c = Buffer::<C64>::alloc(total).unwrap();
        let mut d_buf_c = buf_c.to_device().unwrap();

        let threads_per_block = 256;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = Instant::now();
        let mut d_width = n.to_device().unwrap();
        let mut d_height = n.to_device().unwrap();
        match matrix_mul.launch_with_dptr(
            threads_per_block,
            blocks,
            &mut d_buf_c,
            &mut d_a,
            &mut d_b,
            &mut d_width,
            &mut d_height
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

    println!("Bench 7 Results:");
    for result in &results {
        println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    }

    results
}

pub fn bench8() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();

    let n = 512usize;
    let total = n * n;
    let iters = [1, 10, 20, 50, 100, 200];

    // Generate deterministic pseudo-random matrix data
    let mut a = vec![C64::new(0.0, 0.0); n * n];
    let mut b = vec![C64::new(0.0, 0.0); n * n];
    for i in 0..n * n {
        a[i] = C64::new(i as f64 * 12345.6789, 0.0);
        b[i] = C64::new(0.0, i as f64 * 98765.4321);
    }

    let mut d_a = a.as_slice().to_device().unwrap();
    let mut d_b = b.as_slice().to_device().unwrap();
    let mut d_buf_c = Buffer::<C64>::alloc(total).unwrap().to_device().unwrap();

    let threads_per_block = 256;
    let blocks = (total + threads_per_block - 1) / threads_per_block;

    for &iter in &iters {
        let start = Instant::now();
        let mut d_width = n.to_device().unwrap();
        let mut d_height = n.to_device().unwrap();
        for _ in 0..iter {
            match matrix_mul.launch_with_dptr(
                threads_per_block,
                blocks,
                &mut d_buf_c,
                &mut d_a,
                &mut d_b,
                &mut d_width,
                &mut d_height
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

    println!("\nBench 8 Results:");
    for result in &results {
        println!("Test: {} | {} iterations | GPU Time: {:.3} ms", result.test, result.size, result.gpu_time);
    }

    results
}