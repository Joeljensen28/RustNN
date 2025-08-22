mod layer;
mod utils;
mod activations;
mod loss_functions;
mod datasets;
mod optimizers;

use std::cmp::max;

use ndarray_linalg::krylov::R;
use rand_distr::StandardNormal;
use ndarray::{Array2, Array1, array, Axis, s};
use ndarray_rand::RandomExt;

use crate::{
    activations::{ReLU, Softmax}, datasets::{spiral_data, vertical_data}, layer::Layer, loss_functions::{CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy}, optimizers::SGD, utils::{accuracy, diagflat}
};

fn main() {
    let (x, y) = spiral_data(100, 3);
    let batch_size = x.dim().0;
    let mut dense1 = Layer::new(2, 64, batch_size);
    let mut activation1 = ReLU::new();
    let mut dense2 = Layer::new(64, 3, batch_size);
    let mut loss_activation = SoftmaxCategoricalCrossEntropy::new();
    let optimizer = SGD::new(1.0);

    for _epoch in 0..10001 {
        dense1.forward(&x);
        activation1.forward(&dense1.outputs);
        dense2.forward(activation1.outputs());
        let loss = loss_activation.forward_sparse(&dense2.outputs, &y);
        println!("Loss: {}\n", loss);

        let accuracy = accuracy(loss_activation.outputs(), &y);
        println!("Accuracy: {}\n", accuracy);

        loss_activation.backward_sparse(&loss_activation.outputs().clone(), &y);
        dense2.backward(loss_activation.dinputs());
        activation1.backward(dense2.dinputs());
        dense1.backward(activation1.dinputs());

        optimizer.update_params(dense1); // <--\
                                                //     --------- These are swallowing the layers making the loop not work. Figure out how to update layer params without swallowing layers.
        optimizer.update_params(dense2); // <--/
    }
}
