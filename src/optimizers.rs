use std::f64::EPSILON;

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