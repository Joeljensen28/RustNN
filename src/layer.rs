#![allow(dead_code)]

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub outputs: Array2<f64>,

    pub inputs: Option<Array2<f64>>,
    pub dweights: Option<Array2<f64>>,
    pub dbiases: Option<Array1<f64>>,
    pub dinputs: Option<Array2<f64>>
}

impl Layer {
    pub fn new(n_inputs: usize, n_neurons: usize, batch_size: usize) -> Self {
        let weights = 0.1 * Array2::random((n_inputs, n_neurons), StandardNormal);
        let biases = Array1::zeros(n_neurons);
        let outputs = Array2::zeros((batch_size, n_neurons));

        Layer { 
            weights, 
            biases, 
            outputs, 
            inputs: None, 
            dweights: None, 
            dbiases: None, 
            dinputs: None 
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = Some(inputs.clone());
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        let x = self.inputs.as_ref().expect("No input set. Call `forward` before `backward`.");
        self.dweights = Some(x.t().dot(dvalues));
        self.dbiases = Some(dvalues.sum_axis(Axis(0)));
        self.dinputs = Some(dvalues.dot(&self.weights.t()));
    }

    pub fn inputs(&self) -> &Array2<f64> {
        self.inputs.as_ref().expect("No input set. Make sure to call `forward` first.")
    }
}