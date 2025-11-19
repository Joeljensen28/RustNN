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
    activations::{Linear, ReLU, Sigmoid, Softmax}, 
    datasets::{sine_data, spiral_data, vertical_data}, 
    layers::{Dense, Dropout}, 
    loss_functions::{
        BinaryCrossEntropy, 
        CategoricalCrossEntropy, 
        Loss, 
        MeanSquaredError, 
        SoftmaxCategoricalCrossEntropy
    }, 
    optimizers::{AdaGrad, Adam, RMSProp, SGD}, 
    utils::{Arrayusize, accuracy, accuracy_bce, binarize, diagflat, epsilon_accuracy, std}
};

fn main() {
    let (x, y) = sine_data(1000); // 1000 is the default in nnfs
    let batch_size = x.dim().0;

    let mut dense1 = Dense::new(1, 64, batch_size);
    let mut activation1 = ReLU::new();
    let mut dense2 = Dense::new(64, 64, batch_size);
    let mut activation2 = ReLU::new();
    let mut dense3 = Dense::new(64, 1, batch_size);
    let mut activation3 = Linear::new();
    let mut loss_function = MeanSquaredError::new();

    let mut optimizer = Adam::new();
    optimizer.set_hyperparams(hashmap! {
        "learning_rate" => 0.005,
        "decay" => 1e-3
    });
    let acc_prec = std(&y) / 250.0;

    let mut data_loss;
    let mut reg_loss;
    let mut loss;
    let mut acc;
    for _epoch in 0..10001 {
        dense1.forward(&x);
        activation1.forward(&dense1.outputs);
        dense2.forward(activation1.outputs());
        activation2.forward(&dense2.outputs);
        dense3.forward(activation2.outputs());
        activation3.forward(&dense3.outputs);

        data_loss = loss_function.calculate(activation3.output(), &y);
        reg_loss = 
            loss_function.regularization_loss(&dense1) +
            loss_function.regularization_loss(&dense2) +
            loss_function.regularization_loss(&dense3);
        loss = data_loss + reg_loss;

        let predictions = activation3.output();
        acc = epsilon_accuracy(&predictions, &y, acc_prec.clone());

        if _epoch % 100 == 0 {
            println!("epoch: {}", _epoch);
            println!("acc: {}", acc);
            println!("loss: {}", loss);
            println!("data loss: {}", data_loss);
            println!("reg loss: {}", reg_loss);
            println!("lr: {}\n", optimizer.current_learning_rate);
        }

        loss_function.backward(activation3.output(), &y);
        activation3.backward(loss_function.dinputs());
        dense3.backward(activation3.dinputs());
        activation2.backward(dense3.dinputs());
        dense2.backward(activation2.dinputs());
        activation1.backward(dense2.dinputs());
        dense1.backward(activation1.dinputs());

        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.update_params(&mut dense3);
        optimizer.post_update_params();
    }
}
