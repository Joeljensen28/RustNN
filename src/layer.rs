#![allow(dead_code)]

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub outputs: Array2<f64>,

    pub inputs: Option<Array2<f64>>,
    pub dweights: Option<Array2<f64>>,
    pub dbiases: Option<Array1<f64>>,
    pub dinputs: Option<Array2<f64>>,

    pub weight_momentums: Option<Array2<f64>>,
    pub bias_momentums: Option<Array1<f64>>,

    pub weight_cache: Option<Array2<f64>>,
    pub bias_cache: Option<Array1<f64>>
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
            dinputs: None,
            weight_momentums: None,
            bias_momentums: None,
            weight_cache: None,
            bias_cache: None
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

    pub fn dweights(&self) -> &Array2<f64> {
        self.dweights.as_ref().expect("dweights not yet set. Make sure to call `backward` first.")
    }

    pub fn dbiases(&self) -> &Array1<f64> {
        self.dbiases.as_ref().expect("dbiases not yet set. Make sure to call `backward` first.")
    }

    pub fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("dbdinputs not yet set. Make sure to call `backward` first.")
    }

    pub fn weight_momentums(&self) -> &Array2<f64> {
        self.weight_momentums.as_ref().expect("weights_momentum not yet set. Make sure to update layer params first.")
    }

    pub fn bias_momentums(&self) -> &Array1<f64> {
        self.bias_momentums.as_ref().expect("bias_momentum not yet set. Make sure to update layer params first.")
    }

    pub fn weight_cache(&self) -> &Array2<f64> {
        self.weight_cache.as_ref().expect("weight_cache not yet set. Make sure to update layer params with a weight-caching optimizer first.")
    }

    pub fn bias_cache(&self) -> &Array1<f64> {
        self.bias_cache.as_ref().expect("bias_cache not yet set. Make sure to update layer params with a weight-caching optimizer first.")
    }
}