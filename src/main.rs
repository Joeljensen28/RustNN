mod layer;
mod utils;
mod activations;
mod loss_functions;
mod datasets;

use std::cmp::max;

use rand_distr::StandardNormal;
use ndarray::{Array2, Array1, array, Axis, s};
use ndarray_rand::RandomExt;

use crate::{
    datasets::{spiral_data, vertical_data}, 
    loss_functions::{CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy}, 
    utils::{accuracy, diagflat},
    layer::Layer,
    activations::{ReLU, Softmax}
};

fn main() {
    let (x, y) = spiral_data(100, 3);
    let batch_size = x.dim().1;

    let mut dense1 = Layer::new(2, 3, batch_size);
    let mut activation1 = ReLU::new();
    let mut dense2 = Layer::new(3, 3, batch_size);
    let mut loss_activation = SoftmaxCategoricalCrossEntropy::new();

    dense1.forward(&x);
    activation1.forward(&dense1.outputs);
    dense2.forward(&activation1.outputs());
    let loss = loss_activation.forward_sparse(&dense2.outputs, &y);
    
    println!("loss_activation output: {}\n", loss_activation.outputs().slice(s![0..5, ..]));
    println!("loss: {}\n", loss);

    let accuracy = accuracy(loss_activation.outputs(), &y);
    println!("acc: {}\n", accuracy);

    loss_activation.backward_sparse(&loss_activation.outputs().clone(), &y);
    dense2.backward(loss_activation.dinputs());
    activation1.backward(loss_activation.dinputs());
    dense1.backward(activation1.dinputs());

    println!("dense1.dweights: {:?}\n", dense1.dweights);
    println!("dense1.dbiases: {:?}\n", dense1.dbiases);
    println!("dense2.dweights: {:?}\n", dense2.dweights);
    println!("dense2.dbiases: {:?}\n", dense2.dbiases);
}
