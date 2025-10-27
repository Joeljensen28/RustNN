#![allow(dead_code)]

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis, Array};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use crate::utils::binomial;

pub struct Dense {
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
    pub bias_cache: Option<Array1<f64>>,

    pub weight_regularizer_l1: f64,
    pub weight_regularizer_l2: f64,
    pub bias_regularizer_l1: f64,
    pub bias_regularizer_l2: f64
}

impl Dense {
    pub fn new(n_inputs: usize, n_neurons: usize, batch_size: usize) -> Self {
        let weights = 0.1 * Array2::random(
            (n_inputs, n_neurons), StandardNormal
        );
        let biases = Array1::zeros(n_neurons);
        let outputs = Array2::zeros((batch_size, n_neurons));

        Dense { 
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
            bias_cache: None,
            weight_regularizer_l1: 0.0,
            weight_regularizer_l2: 0.0,
            bias_regularizer_l1: 0.0,
            bias_regularizer_l2: 0.0
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = Some(inputs.clone());
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        let x = self.inputs
            .as_ref()
            .expect("No input set. Call `forward` before `backward`.");
        self.dweights = Some(x.t().dot(dvalues));
        self.dbiases = Some(dvalues.sum_axis(Axis(0)));

        if self.weight_regularizer_l1 > 0.0 {
            let mut dl1: Array2<f64> = Array::ones(self.weights.dim());
            dl1.zip_mut_with(
                &self.weights, 
                |dl1_v, &weight| if weight < 0.0 { *dl1_v = -1.0 }
            );
            let dweights = self.dweights
                .as_ref()
                .expect("dweights unexpectedly empty.");
            self.dweights = Some(dweights + (self.weight_regularizer_l1 * dl1));
        }

        if self.weight_regularizer_l2 > 0.0 {
            let dweights = self.dweights
                .as_ref()
                .expect("dweights unexpectedly empty.");
            self.dweights = Some(dweights + (2.0 * &self.weight_regularizer_l2 * &self.weights));
        }

        if self.bias_regularizer_l1 > 0.0 {
            let mut dl1: Array1<f64> = Array::ones(self.biases.dim());
            dl1.zip_mut_with(
                &self.biases, 
                |dl1_v, &bias| if bias < 0.0 { *dl1_v = -1.0 }
            );
            let dbiases = self.dbiases.as_ref().expect("dbiases unexpectedly empty.");
            self.dbiases = Some(dbiases + (self.bias_regularizer_l1 * dl1));
        }

        if self.bias_regularizer_l2 > 0.0 {
            let dbiases = self.dbiases.as_ref().expect("dbiases unexpectedly empty.");
            self.dbiases = Some(dbiases + (2.0 * &self.bias_regularizer_l2 * &self.biases));
        }

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
        self.weight_cache
            .as_ref()
            .expect(
                "weight_cache not yet set. Make sure to update layer params with a weight-caching optimizer first."
            )
    }

    pub fn bias_cache(&self) -> &Array1<f64> {
        self.bias_cache
            .as_ref()
            .expect("bias_cache not yet set. Make sure to update layer params with a weight-caching optimizer first.")
    }

    pub fn set_regularizers(&mut self, regs: HashMap<&str, f64>) {
        for (key, value) in regs {
            match key {
                "weight_l1" => self.weight_regularizer_l1 = value,
                "weight_l2" => self.weight_regularizer_l2 = value,
                "bias_l1" => self.bias_regularizer_l1 = value,
                "bias_l2" => self.bias_regularizer_l2 = value,
                _ => panic!(
                        "Invalid hyperparamter \"{}\" passed.
                        Valid hyperparameters:\n\tweight_l1,\n\tweight_l2,\n\tbias_l1,\n\tbias_l2", key
                    )
            }
        }
    }
}

pub struct Dropout {
    pub rate: f64,
    pub inputs: Option<Array2<f64>>,
    pub binary_mask: Option<Array2<f64>>,
    pub output: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Dropout { 
            rate: rate, 
            inputs: None, 
            binary_mask: None, 
            output: None, 
            dinputs: None
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = Some(inputs.clone());
        self.binary_mask = Some(
            binomial(1, self.rate, inputs.dim()) / self.rate
        );
        self.output = Some(inputs * self.binary_mask.as_ref().expect("self.binary_mask unexpectedly empty"));
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        self.dinputs = Some(
            dvalues * self.binary_mask.as_ref().expect("self.binary_mask is empty. Did you call .forward() first?")
        );
    }

    pub fn outputs(&self) -> &Array2<f64> {
        self.output.as_ref().expect("output not yet set. Be sure to call .forward() first.")
    }

    pub fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("dinputs not yet set. Be sure to call .backward() first.")
    }
}