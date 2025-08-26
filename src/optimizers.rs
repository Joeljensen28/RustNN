use crate::layer::Layer;

pub struct SGD {
    pub learning_rate: f64,
    pub decay: f64,
    pub current_learning_rate: f64,
    pub iterations: i64
}

impl SGD {
    pub fn new(learning_rate: f64, decay: f64) -> Self {
        SGD{ 
            learning_rate: learning_rate,
            decay: decay,
            current_learning_rate: learning_rate,
            iterations: 0
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
        layer.weights.scaled_add(-self.current_learning_rate, &layer.dweights().clone());
        layer.biases.scaled_add(-self.current_learning_rate, &layer.dbiases().clone());
    }

    pub fn post_update_paramts(&mut self) {
        self.iterations += 1;
    }
}