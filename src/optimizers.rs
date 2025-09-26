use std::{collections::HashMap, f64::EPSILON};

use ndarray::{Array1, Array2};

use crate::layer::Layer;

pub struct SGD {
    pub learning_rate: f64,
    pub decay: f64,
    pub current_learning_rate: f64,
    pub iterations: i64,
    pub momentum: f64
}

impl SGD {
    pub fn new(learning_rate: f64, decay: f64, momentum: f64) -> Self {
        SGD{ 
            learning_rate: learning_rate,
            decay: decay,
            current_learning_rate: learning_rate,
            iterations: 0,
            momentum: momentum
        }
    }

    pub fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * (self.iterations as f64))
            );
        }
    }

    pub fn update_params(&self, layer: &mut Layer) {
        let weight_updates;
        let bias_updates;

        if self.momentum != 0.0 {
            if layer.weight_momentums.is_none() {
                layer.weight_momentums = Some(Array2::zeros(layer.weights.dim()));
                layer.bias_momentums = Some(Array1::zeros(layer.biases.dim()));
            }

            weight_updates = 
                self.momentum * layer.weight_momentums() - self.current_learning_rate * layer.dweights();
            layer.weight_momentums = Some(weight_updates.clone());

            bias_updates = 
                self.momentum * layer.bias_momentums() - self.current_learning_rate * layer.dbiases();
            layer.bias_momentums = Some(bias_updates.clone());
        }

        else {
            weight_updates = -self.current_learning_rate * layer.dweights();
            bias_updates = -self.current_learning_rate * layer.dbiases();
        }

        layer.weights += &weight_updates;
        layer.biases += &bias_updates;
    }

    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

pub struct AdaGrad {
    pub learning_rate: f64,
    pub decay: f64,
    pub current_learning_rate: f64,
    pub iterations: i64,
    pub epsilon: f64
}

impl AdaGrad {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64) -> Self {
        AdaGrad { 
            learning_rate: learning_rate,
            decay: decay,
            current_learning_rate: learning_rate,
            iterations: 0,
            epsilon: epsilon
        }
    }

    pub fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * (self.iterations as f64))
            );
        }
    }

    pub fn update_params(&self, layer: &mut Layer) {
        if layer.weight_cache.is_none() {
            layer.weight_cache = Some(Array2::zeros(layer.weights.dim()));
            layer.bias_cache = Some(Array1::zeros(layer.biases.dim()))
        }

        let mut weight_cache = layer.weight_cache.clone().expect("weight_cache is not yet initialized");
        let mut bias_cache = layer.bias_cache.clone().expect("bias_cache is not yet initialized");
        let weight_pow = layer.dweights().mapv(|x| x.powi(2));
        let bias_pow = layer.dbiases().mapv(|x| x.powi(2));
        weight_cache += &weight_pow;
        bias_cache += &bias_pow;
        layer.weight_cache = Some(weight_cache.clone());
        layer.bias_cache = Some(bias_cache.clone());

        layer.weights += &(-self.current_learning_rate * layer.dweights() / (weight_cache.mapv(|x| x.sqrt()) + self.epsilon));
        layer.biases += &(-self.current_learning_rate * layer.dbiases() / (bias_cache.mapv(|x| x.sqrt()) + self.epsilon));
    }

    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

pub struct RMSProp {
    pub learning_rate: f64,
    pub decay: f64,
    pub current_learning_rate: f64,
    pub iterations: i64,
    pub epsilon: f64,
    pub rho: f64
}

impl RMSProp {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64, rho: f64) -> Self {
        RMSProp {
            learning_rate: learning_rate,
            decay: decay,
            iterations: 0,
            current_learning_rate: learning_rate,
            epsilon: epsilon,
            rho: rho
        }
    }

    pub fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * (self.iterations as f64))
            );
        }
    }

    pub fn update_params(&mut self, layer: &mut Layer) {
        if layer.weight_cache.is_none() {
            layer.weight_cache = Some(Array2::zeros(layer.weights.dim()));
            layer.bias_cache = Some(Array1::zeros(layer.biases.dim()))
        }

        let mut weight_cache = layer.weight_cache.clone().expect("weight_cache not yet initialized");
        let mut bias_cache = layer.bias_cache.clone().expect("bias_cache not yet initialized");
        weight_cache = self.rho * weight_cache + (1.0 - self.rho) * layer.dweights().mapv(|x| x.powi(2));
        bias_cache = self.rho * bias_cache + (1.0 - self.rho) * layer.dbiases().mapv(|x| x.powi(2));
        layer.weight_cache = Some(weight_cache.clone());
        layer.bias_cache = Some(bias_cache.clone());

        layer.weights += &(-self.current_learning_rate * layer.dweights() / (weight_cache.mapv(|x| x.sqrt() + self.epsilon)));
        layer.biases += &(-self.current_learning_rate * layer.dbiases() / bias_cache.mapv(|x| x.sqrt() + self.epsilon));
    }

    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

pub struct Adam {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iterations: i64,
    epsilon: f64,
    beta_1: f64,
    beta_2: f64
}

impl Adam {
    pub fn new() -> Self {
        Adam {
            learning_rate: 0.001,
            current_learning_rate: 0.001,
            decay: 0.0,
            iterations: 0,
            epsilon: 1e-7,
            beta_1: 0.9,
            beta_2: 0.999
        }
    }

    pub fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * (self.iterations as f64))
            );
        }
    }

    pub fn update_params(&mut self, layer: &mut Layer) {
        if layer.weight_cache.is_none() {
            layer.weight_momentums = Some(Array2::zeros(layer.weights.dim()));
            layer.weight_cache = Some(Array2::zeros(layer.weights.dim()));
            layer.bias_momentums = Some(Array1::zeros(layer.biases.dim()));
            layer.bias_cache = Some(Array1::zeros(layer.biases.dim()));
        }

        layer.weight_momentums = Some(self.beta_1 * layer.weight_momentums() + (1.0 - self.beta_1) * layer.dweights());
        layer.bias_momentums = Some(self.beta_1 * layer.bias_momentums() + (1.0 - self.beta_1) * layer.dbiases());

        let weight_momentums_corrected 
            = layer.weight_momentums() / (1.0 - self.beta_1.powi(self.iterations as i32 + 1));
        let bias_momentums_corrected
            = layer.bias_momentums() / (1.0 - self.beta_1.powi(self.iterations as i32 + 1));
        
        layer.weight_cache = Some(
            self.beta_2 * layer.weight_cache() + (1.0 - self.beta_2) * layer.dweights().mapv(|x| x.powi(2))
        );
        layer.bias_cache = Some(
            self.beta_2 * layer.bias_cache() + (1.0 - self.beta_2) * layer.dbiases().mapv(|x| x.powi(2))
        );

        let weight_cache_corrected 
            = layer.weight_cache() / (1.0 - self.beta_2.powi(self.iterations as i32 + 1));
        let bias_cahce_corrected
            = layer.bias_cache() / (1.0 - self.beta_2.powi(self.iterations as i32 + 1));

        layer.weights += 
            &(-self.current_learning_rate * weight_momentums_corrected / 
            (weight_cache_corrected.mapv(|x| x.sqrt()) + self.epsilon));
        layer.biases +=
            &(-self.current_learning_rate * bias_momentums_corrected / 
            (bias_cahce_corrected.mapv(|x| x.sqrt()) + self.epsilon));
    }

    pub fn post_update_params(&mut self) {
        self.iterations += 1;
    }

    pub fn set_hyperparams(&mut self, hyperparams: HashMap<&str, f64>) {
        for (key, value) in hyperparams {
            match key {
                "learning_rate" => self.learning_rate = value,
                "decay" => self.decay = value,
                "epsilon" => self.epsilon = value,
                "beta_1" => self.beta_1 = value,
                "beta_2" => self.beta_2 = value,
                _ => panic!("Invalid hyperparamter \"{}\" passed.", key)
            }
        }
    }
}