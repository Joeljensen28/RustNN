mod layers;
mod utils;
mod activations;
mod loss_functions;
mod datasets;
mod optimizers;

use std::{backtrace, cmp::max, collections::HashMap};
use maplit::hashmap;

use ndarray_linalg::krylov::R;
use rand_distr::StandardNormal;
use ndarray::{Array2, Array1, array, Axis, s};
use ndarray_rand::RandomExt;

use crate::{
    activations::{ReLU, Sigmoid, Softmax}, 
    datasets::{spiral_data, vertical_data}, 
    layers::{Dense, Dropout}, 
    loss_functions::{BinaryCrossEntropy, CategoricalCrossEntropy, Loss, SoftmaxCategoricalCrossEntropy}, 
    optimizers::{AdaGrad, Adam, RMSProp, SGD}, 
    utils::{Arrayusize, accuracy, binarize, diagflat, accuracy_bce}
};

fn main() {
    let (x, y) = spiral_data(100, 2);
    let y = y.clone().into_shape((y.len(), 1)).expect("reshape failed");

    let batch_size = x.dim().0;
    
    let mut dense1 = Dense::new(2, 64, batch_size);
    dense1.set_regularizers(hashmap! {
        "weight_l2" => 5e-4,
        "bias_l2" => 5e-4
    });
    let mut activation1 = ReLU::new();
    let mut dense2 = Dense::new(64, 1, batch_size);
    let mut activation2 = Sigmoid::new();
    let mut loss_function = BinaryCrossEntropy::new();
    let mut optimizer = Adam::new();
    optimizer.set_hyperparams(hashmap! {
        "decay" => 5e-7
    });

    let mut data_loss;
    let mut acc;
    let mut reg_loss;
    let mut loss;
    let mut predictions;

    for epoch in 0..10001 {
        dense1.forward(&x);
        activation1.forward(&dense1.outputs);
        dense2.forward(activation1.outputs());
        activation2.forward(&dense2.outputs);

        data_loss = loss_function.calculate(activation2.outputs(), Arrayusize::Array2(&y));
        reg_loss = loss_function.regularization_loss(&dense1) +
            loss_function.regularization_loss(&dense2);

        loss = data_loss + reg_loss;

        predictions = binarize(activation2.outputs(), 0.5);
        acc = accuracy_bce(&predictions, &y);

        if epoch % 100 == 0 {
            println!("epoch: {}", epoch);
            println!("acc: {}", acc);
            println!("loss: {}", loss);
            println!("data_loss: {}", data_loss);
            println!("reg_loss: {}", reg_loss);
            println!("lr: {}\n", optimizer.current_learning_rate);
        }

        loss_function.backward(activation2.outputs(), Arrayusize::Array2(&y));
        activation2.backward(loss_function.dinputs());
        dense2.backward(activation2.dinputs());
        activation1.backward(dense2.dinputs());
        dense1.backward(activation1.dinputs());

        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.post_update_params();
    }

    let (x_test, y_test) = spiral_data(100, 2);
    let y_test = y_test.clone().into_shape((y.len(), 1)).expect("reshape failed");

    dense1.forward(&x_test);
    activation1.forward(&dense1.outputs);
    dense2.forward(activation1.outputs());
    activation2.forward(&dense2.outputs);
    
    let loss = loss_function.calculate(activation2.outputs(), Arrayusize::Array2(&y_test));
    let predictions = binarize(activation2.outputs(), 0.5);
    let acc = accuracy_bce(&predictions, &y_test);

    println!("validation acc: {}", acc);
    println!("loss: {}", loss);
}
