use cuda::dmem::{Buffer, DSend, DeepCopy};
use cuda::gpu;
use cuda::device_sync;
use std::time::Instant;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};
use num::complex::Complex;
use rustfft::FftPlanner;
use crate::BenchResults;

#[derive(Debug, Clone, Copy)]
struct C64 { re: f64, im: f64 }

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }

    fn exp(self) -> Self {
        let exp_x = self.re.exp();
        C64 {
            re: exp_x * self.im.cos(),
            im: exp_x * self.im.sin(),
        }
    }
}

impl DeepCopy for C64 {

}

impl Add for C64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        C64 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}
impl Sub for C64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        C64 {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}
impl Mul for C64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        C64 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[kernel]
unsafe fn matrix_mul(mut result: Buffer<C64>,
                     a: &[C64], b: &[C64], width: usize, height: usize) {
    let tid = gpu::global_tid_x() as usize;
    let i = tid / height;
    let j = tid % height;
    let mut sum:C64 = C64::new(0.0, 0.0);
    for k in 0..width {
        sum = sum + a[i * width + k] * b[k * height + j];
    }
    result.set(i * height + j, sum);
}

#[kernel]
unsafe fn fft_kernel(mut result: Buffer<C64>,
                     x_input: Buffer<C64>,
                     width: usize, height: usize) {
    let tid = gpu::global_tid_x() as usize;
    let i = tid / (height >> 1);
    let j = tid % (height >> 1);
    let temp = C64::new(0.0, -1.0 * PI * (i as f64) / (width as f64)
    ).exp() * x_input.get(i * height + j + (height >> 1));

    result.set(tid as usize, x_input.get(i * height + j) + temp);
    result.set((i+width) * (height >> 1) + j, x_input.get(i * height + j) - temp);
}

fn fast_fourier_transform(x: &[C64], mut width: usize) -> (Vec<C64>, Vec<BenchResults>) {
    let n = x.len();
    let mut results = Vec::new();

    if n.count_ones() != 1 {
        panic!("size of x must be a power of 2");
    }

    // Precompile the kernels.
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }
    match fft_kernel.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut height = n / width;

    let new_x_buffer: Buffer<C64> = Buffer::alloc(n).unwrap();
    let mut matrix = vec![C64::new(0.0, 0.0); width * width];
    for i in 0..width {
        for j in 0..width {
            matrix[i * width + j] = C64::new(0.0, -2.0 *
                PI * (i as f64) * (j as f64) / (width as f64)).exp();
        }
    }

    let mut d_new_x_buffer = new_x_buffer.to_device().unwrap();
    let mut d_matrix = matrix.as_slice().to_device().unwrap();
    let mut d_x = x.to_device().unwrap();

    let threads_per_block = 256;
    let blocks = n / threads_per_block;

    let start = Instant::now();
    let mut d_width = width.to_device().unwrap();
    let mut d_height = height.to_device().unwrap();
    match matrix_mul.launch_with_dptr(
        threads_per_block as usize,
        blocks as usize,
        &mut d_new_x_buffer,
        &mut d_matrix,
        &mut d_x,
        &mut d_width,
        &mut d_height,
    ) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error launching matrix multiplication kernel: {:?}", e);
        }
    }
    match device_sync() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error synchronizing device after matrix multiplication kernel: {:?}", e);
        }
    }
    let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchResults {
        gpu_time: elapsed_gpu_ms,
        test: format!("{}x{}", width, height),
        size: 0,
    });

    // Bit shift one to the right == division by 2.
    let blocks = blocks >> 1;
    let mut counter = 1;

    // First buffer is empty.
    let buffer_1: Buffer<C64> = Buffer::alloc(n).unwrap();
    let mut d_buffer_1 = buffer_1.to_device().unwrap();
    // Second buffer is the result from the previous calculation.
    let mut d_buffer_2 = d_new_x_buffer;

    while width < n {
        let start = Instant::now();

        d_width = width.to_device().unwrap();
        d_height = height.to_device().unwrap();
        match fft_kernel.launch_with_dptr(
            threads_per_block as usize,
            blocks as usize,
            &mut d_buffer_1,
            &mut d_buffer_2,
            &mut d_width,
            &mut d_height,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error launching fft kernel: {:?}", e);
            }
        }
        match device_sync() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error synchronizing device after fft kernel: {:?}", e);
            }
        }
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        let bench_results = BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("{}x{}", width, height),
            size: counter,
        };
        results.push(bench_results.clone());
        println!("Test: {}_iteration_{} | GPU Time: {:.3} ms", bench_results.test, bench_results.size, bench_results.gpu_time);

        // Switch the buffers.
        (d_buffer_1, d_buffer_2) = (d_buffer_2, d_buffer_1);
        
        width <<= 1;
        height >>= 1;
        counter += 1;
    }

    return (new_x_buffer.retrieve().unwrap(), results);
}


pub fn bench5() -> Vec<BenchResults> {
    let n = 8192*16*16;

    let mut complex_pseudo_random_input = vec![Complex::new(0.0, 0.0); n];
    let mut pseudo_random_input = vec![C64::new(0.0, 0.0); n];

    for i in 0..n {
        let input_value = i as f64 * 12345.6789;

        complex_pseudo_random_input[i] = Complex::new(input_value, 0.0);
        pseudo_random_input[i] = C64::new(input_value, 0.0);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut complex_pseudo_random_input);

    let output = fast_fourier_transform(&pseudo_random_input, 32);

    let mut max_distance = 0.0;
    for (point1, point2) in output.0.iter().zip(complex_pseudo_random_input.iter()) {
        let dx = point1.re - point2.re;
        let dy = point1.im - point2.im;
        let distance = (dx * dx + dy * dy).sqrt();
        if distance > max_distance {
            max_distance = distance;
        }
    }
    println!("Maximum distance from rust.fft: {}", max_distance);

    return output.1
}