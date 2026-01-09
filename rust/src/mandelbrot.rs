use cuda::dmem::{Buffer};
use cuda::gpu;
use image::ColorType;
use std::path::Path;
use std::time::Instant;
use crate::BenchResults;

// Adapted from https://github.com/NiekAukes/Gendelbrot,
// which is forked from https://github.com/JohnTWilkinson/Gendelbrot

// Simple struct for complex numbers
#[derive(Debug, Clone)]
pub struct Complex {
    real: f64,
    imaginary: f64,
}

// The functions below execute various calculations according to how a mandelbrot is generated,
// a full description would take several paragraphs, so if you want a full explanation of what
// is happening in the functions below, consider watching this video:
// https://www.youtube.com/watch?v=FFftmWSzgmk
impl Complex {
    // Iterates the complex number once using the mandelbrot algorithm
    fn iterate(&mut self, origin: &Complex) {
        let copy = self.clone();
        self.real = (copy.real * copy.real) - (copy.imaginary * copy.imaginary) + origin.real;
        self.imaginary = (copy.real + copy.real) * copy.imaginary + origin.imaginary;
    }

    // Checks to see if the complex number has gone past the escape radius
    fn has_escaped(&self) -> bool {
        self.real * self.real + self.imaginary * self.imaginary >= 4.0
    }

    // Returns a new complex number
    fn new(x: f64, y: f64) -> Complex {
        Complex {
            real: x,
            imaginary: y,
        }
    }

    // Runs the complete mandelbrot algorithm and returns on a scale from 0-255 how many iterations
    // it took for the values to escape. (255 for values that do not go past the escape radius)
    // The complex number will be iterated a maximum of
    // {stable_iterations} times before the algorithm decides it's in the mandelbrot set,
    // assuming it doesn't escape before then.
    fn mandel(&self, stable_iterations: i32) -> u8 {
        let mut copy: Complex = self.clone();
        for i in 0..stable_iterations {
            if copy.has_escaped() {
                let color: i32 = (255 * i) / stable_iterations;
                return color as u8;
            }
            copy.iterate(self);
        }
        return 255
    }
}

#[kernel]
fn create_mandelbrot_fractal(
    mut image: Buffer<u8>,
    pixel_span: usize,
    coordinate_span: f64,
    real_start: f64,
    i_start: f64,
    max_iterations: i32,
) {
    let tid = gpu::global_tid_x() as usize;
    let x_pixel = tid / pixel_span;
    let y_pixel = tid % pixel_span;
    let step = coordinate_span / pixel_span as f64;

    if x_pixel < pixel_span && y_pixel < pixel_span {
        let x = real_start + (x_pixel as f64 * step);
        let y = i_start + (y_pixel as f64 * step);
        let c = Complex::new(x, y);
        image.set(y_pixel * pixel_span + x_pixel, c.mandel(max_iterations));
    };
}

pub fn bench3() -> Vec<BenchResults> {
    // 1 is for precompilation
    let max_iterations_list = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let mut results = Vec::new();

    let pixel_span: usize = 8192;
    let image_buffer: Buffer<u8> = Buffer::alloc(pixel_span*pixel_span).unwrap();

    let threads_per_block = 256;
    let blocks = (pixel_span*pixel_span + threads_per_block - 1) / threads_per_block;

    for max_iterations in max_iterations_list {
        let start = Instant::now();
        match create_mandelbrot_fractal.launch(
            threads_per_block as usize,
            blocks as usize,
            image_buffer,
            pixel_span,
            3.0,
            -2.0,
            -1.5,
            max_iterations,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error launching kernel: {:?}", e);
            }
        }
        image_buffer.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        if max_iterations != 1 {
            let bench_results = BenchResults {
                gpu_time: elapsed_gpu_ms,
                test: format!("{}x{}", pixel_span, pixel_span),
                size: max_iterations as usize,
            };
            results.push(bench_results.clone());
            println!("Test: {} | {} max_iters | GPU Time: {:.3} ms", bench_results.test, bench_results.size, bench_results.gpu_time);
        }
    }
    return results;
}

pub fn bench4() -> Vec<BenchResults> {
    // 1 is for precompilation
    let iterations_list = [1, 10, 20, 50, 100];
    let max_iterations = 64;
    let mut image_result = Vec::new();
    let mut results = Vec::new();

    let pixel_span: usize = 8192;
    let image_buffer: Buffer<u8> = Buffer::alloc(pixel_span*pixel_span).unwrap();

    let threads_per_block = 256;
    let blocks = (pixel_span*pixel_span + threads_per_block - 1) / threads_per_block;

    for iterations in iterations_list {
        let start = Instant::now();
        for _ in 0..iterations {
            match create_mandelbrot_fractal.launch(
                threads_per_block as usize,
                blocks as usize,
                image_buffer,
                pixel_span,
                3.0,
                -2.0,
                -1.5,
                max_iterations,
            ) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error launching kernel: {:?}", e);
                }
            }

            image_result = image_buffer.retrieve().unwrap();
        }
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        if iterations != 1 {
            let bench_results = BenchResults {
                gpu_time: elapsed_gpu_ms,
                test: format!("{}x{}_max_iterations_{}", pixel_span, pixel_span, max_iterations),
                size: iterations as usize,
            };
            results.push(bench_results.clone());
            println!("Test: {} | iterations: {} | GPU Time: {:.3} ms", bench_results.test, bench_results.size, bench_results.gpu_time);
        }
    }

    let image_path = Path::new("./results/mandelbrot.png");

    // Write the image contents to a file (format automatically deduced from filename)
    image::save_buffer(
        image_path,
        &image_result,
        pixel_span as u32,
        pixel_span as u32,
        ColorType::L8,
    ).expect("Couldn't create or overwrite file!");

    // Done! (image files close automatically when dropped)
    println!(
        "\nDone. File outputted to {:?}",
        dunce::canonicalize(Path::new("./results/mandelbrot.png")).unwrap()
    );

    return results;
}