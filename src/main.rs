mod layer;
mod utils;
mod activations;
mod loss_functions;
mod datasets;
mod optimizers;

use std::{backtrace, cmp::max};

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
    let mut optimizer = SGD::new(1.0, 1e-3);

    let mut loss = f64::MAX;
    let mut acc = f64::MIN;

    for _epoch in 0..10001 {
        dense1.forward(&x);
        activation1.forward(&dense1.outputs);
        dense2.forward(activation1.outputs());
        loss = loss_activation.forward_sparse(&dense2.outputs, &y);
        acc = accuracy(loss_activation.outputs(), &y);

        if _epoch % 100 == 0 {
            println!("epoch: {}", _epoch);
            println!("acc: {}", acc);
            println!("loss: {}", loss);
            println!("lr: {}", optimizer.current_learning_rate);
        }

        loss_activation.backward_sparse(&loss_activation.outputs().clone(), &y);
        dense2.backward(loss_activation.dinputs());
        activation1.backward(dense2.dinputs());
        dense1.backward(activation1.dinputs());

        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.post_update_paramts();
    }
}
