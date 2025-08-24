#![allow(dead_code)]

use ndarray::{Array1, Array2};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use crate::utils::linspace;

pub fn spiral_data(samples: usize, classes: usize) -> (Array2<f64>, Array1<usize>) {
    let total = samples * classes;
    let mut x = Array2::<f64>::zeros((total, 2));
    let mut y = Array1::<usize>::zeros(total);

    let noise = Normal::new(0.0, 0.2).unwrap();
    let mut rng = thread_rng();

    for class_number in 0..classes {
        let base_idx = class_number * samples;

        let r = linspace(0.0, 1.0, samples);

        let t: Vec<f64> = linspace(
            class_number as f64 * 4.0,
            (class_number + 1) as f64 * 4.0,
            samples,
        )
        .into_iter()
        .map(|v| v + noise.sample(&mut rng))
        .collect();

        for i in 0..samples {
            let idx = base_idx + i;
            let theta = t[i] * 2.5;
            x[(idx, 0)] = r[i] * theta.sin();
            x[(idx, 1)] = r[i] * theta.cos();
            y[idx] = class_number as usize;
        }
    }

    (x, y)
}

pub fn vertical_data(samples: usize, classes: usize) -> (Array2<f64>, Array1<usize>) {
    let total = samples * classes;
    let mut x = Array2::<f64>::zeros((total, 2));
    let mut y = Array1::<usize>::zeros(total);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    for class_number in 0..classes {
        let base = class_number * samples;
        for i in 0..samples {
            let idx = base + i;
            let x0 = normal.sample(&mut rng) * 0.1 + (class_number as f64) / 3.0;
            let x1 = normal.sample(&mut rng) * 0.1 + 0.5;
            x[[idx, 0]] = x0;
            x[[idx, 1]] = x1;
            y[idx] = class_number as usize;
        }
    }

    (x, y)
}