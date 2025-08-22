use ndarray::{Array1};

pub struct Perceptron {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl Perceptron {
    pub fn new(weights: Array1<f64>, bias: f64) -> Self {
        Perceptron { weights, bias }
    }

    pub fn activate(&self, inputs: &Array1<f64>) -> f64 {
        self.weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| weight * input)
            .sum::<f64>() + self.bias
    }
}