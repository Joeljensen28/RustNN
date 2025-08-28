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

    pub fn post_update_paramts(&mut self) {
        self.iterations += 1;
    }
}