#![allow(dead_code)]

use ndarray::{Array1, Array2, Axis};

fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num < 2 {
        return vec![stop];
    }
    let step = (stop - start) / (num - 1) as f64;
    (0..num).map(|i| start + step * i as f64).collect()
}

pub fn clip(a: &Array2<f64>, interval_min: f64, interval_max: f64) -> Array2<f64> {
    a.mapv(|x| x.clamp(interval_min, interval_max))
}

pub fn accuracy(y_pred: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
    let n = y_true.len() as f64;

    let pred_classes: Array1<usize> = y_pred
        .axis_iter(Axis(0))
        .map(
            |a| {
                let array_max = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
                a
                    .iter()
                    .position(|&x| x == array_max)
                    .expect("No target found in array")
            }
        )
        .collect();

    let n_correct = pred_classes
        .iter()
        .zip(y_true.iter())
        .filter(
            |(p, t)| p == t
        )
        .count();

    n_correct as f64 / n
}

pub fn to_sparse(one_hot: &Array2<usize>) -> Array1<usize> {
        one_hot
            .axis_iter(Axis(0))
            .map(
                |a| {
                    a
                        .iter()
                        .position(|&x| x == 1)
                        .expect("No target found in array")
                }
            )
            .collect()
    }

pub fn to_one_hot(sparse: Array1<usize>, n_classes: usize) -> Array2<usize> {
    let n_samples = sparse.len();

    Array2::from_shape_fn((n_samples, n_classes), |(i, j)| {
        (sparse[i] == j) as usize
    })
}

pub fn diagflat(a: &Array2<f64>) -> Array2<f64> {
    Array2::eye(a.dim().0) * a
}