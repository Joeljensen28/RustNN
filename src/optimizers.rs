use crate::layer::Layer;

pub struct SGD {
    pub learning_rate: f64
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD{ learning_rate: learning_rate }
    }

    pub fn update_params(&self, layer: &mut Layer) {
        layer.weights.scaled_add(-self.learning_rate, &layer.dweights().clone());
        layer.biases.scaled_add(-self.learning_rate, &layer.dbiases().clone());
    }
}