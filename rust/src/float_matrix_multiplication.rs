use cuda::dmem::{Buffer, DSend, DeepCopy};
use cuda::{gpu};
use std::time::Instant;
use crate::BenchResults;

#[kernel]
unsafe fn matrix_mul(mut result: Buffer<f64>,
                         a: &[f64], b: &[f64], width: usize, height: usize) {
    let tid = gpu::global_tid_x() as usize;
    let i = tid / height;
    let j = tid % height;
    let mut sum:f64 = 0.0;
    for k in 0..width {
        sum = sum + a[i * width + k] * b[k * height + j];
    }
    result.set(i * height + j, sum);
}

#[kernel]
unsafe fn matrix_mul_f32(mut result: Buffer<f32>,
                     a: &[f32], b: &[f32], width: usize, height: usize) {
    let tid = gpu::global_tid_x() as usize;
    let i = tid / height;
    let j = tid % height;
    let mut sum:f32 = 0.0;
    for k in 0..width {
        sum = sum + a[i * width + k] * b[k * height + j];
    }
    result.set(i * height + j, sum);
}

pub fn bench9() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();
    let sizes: [usize;7] = [64, 128, 256, 512, 1024, 2048, 4096];

    for n in sizes {
        let total = n * n;

        // Generate deterministic pseudo-random matrix data
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        for i in 0..n * n {
            a[i] = i as f64 * 12345.6789 % 1000.0;
            b[i] = i as f64 * 98765.4321 % 1000.0;
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let buf_c = Buffer::<f64>::alloc(total).unwrap();
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

    println!("Bench 9 Results:");
    for result in &results {
        println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    }

    results
}

pub fn bench10() -> Vec<BenchResults> {
    match matrix_mul_f32.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();
    let sizes: [usize;7] = [64, 128, 256, 512, 1024, 2048, 4096];

    for n in sizes {
        let total = n * n;

        // Generate deterministic pseudo-random matrix data
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        for i in 0..n * n {
            a[i] = i as f32 * 12345.6789 % 1000.0;
            b[i] = i as f32 * 98765.4321 % 1000.0;
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let buf_c = Buffer::<f32>::alloc(total).unwrap();
        let mut d_buf_c = buf_c.to_device().unwrap();

        let threads_per_block = 256;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = Instant::now();
        let mut d_width = n.to_device().unwrap();
        let mut d_height = n.to_device().unwrap();
        match matrix_mul_f32.launch_with_dptr(
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

    println!("Bench 10 Results:");
    for result in &results {
        println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    }

    results
}