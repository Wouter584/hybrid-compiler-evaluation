use cuda::dmem::{Buffer, DSend};
use cuda::{device_sync, gpu};
use crate::BenchResults;
use std::time::Instant;

#[kernel]
unsafe fn sum_kernel(input_array: &[f32], mut out: Buffer<f32>) {
    let idx = gpu::__nvvm_thread_idx_x() as usize;
    let val = input_array[idx];
// TODO: Add reduction implementation with shared memory
}

pub fn bench6() -> Vec<BenchResults> {
    match sum_kernel.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();

    let size = 256;
    let iterations_list = [1, 10, 100, 1000, 10000, 100000, 1000000];

    // Generate deterministic pseudo-random input
    let mut pseudo_random_input = vec![0f32; size];
    for i in 0..size {
        pseudo_random_input[i] = (i as f32 * 123.0) as f32;
    }
    let expect:f32 = pseudo_random_input.as_slice().iter().sum();

    let mut d_pseudo_random_input = pseudo_random_input.as_slice().to_device().unwrap();
    let mut d_out = Buffer::<f32>::alloc(1).unwrap().to_device().unwrap();

    for iterations in iterations_list {
        let start = Instant::now();
        for _ in 0..iterations {
            match sum_kernel.launch_with_dptr(
                size,
                1usize,
                &mut d_pseudo_random_input,
                &mut d_out,
            ) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error launching atomic_sum_kernel kernel: {:?}", e);
                }
            }
            match device_sync() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error synchronizing device after atomic_sum_kernel kernel: {:?}", e);
                }
            }
        }
        let out = d_out.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("for iterations {:?}, expected: {:?}, got: {:?}", iterations, expect, out.get(0));
        results.push(BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("bench6_sum_result: {:?}", out.get(0)),
            size: iterations,
        });
    }
    return results
}