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
    activations::{ReLU, Softmax}, 
    datasets::{spiral_data, vertical_data}, 
    layers::{Dense, Dropout}, 
    loss_functions::{CategoricalCrossEntropy, Loss, SoftmaxCategoricalCrossEntropy}, 
    optimizers::{AdaGrad, Adam, RMSProp, SGD}, 
    utils::{accuracy, diagflat, Arrayusize}
};

fn main() {
    let (x, y) = spiral_data(1000, 3);
    let batch_size = x.dim().0;
    
    let mut dense1 = Dense::new(2, 512, batch_size);
    dense1.set_regularizers(hashmap! {
        "weight_l2" => 5e-4,
        "bias_l2" => 5e-4
    });
    let mut activation1 = ReLU::new();
    let mut dropout1 = Dropout::new(0.1);
    let mut dense2 = Dense::new(512, 3, batch_size);
    let mut loss_activation = SoftmaxCategoricalCrossEntropy::new();
    let mut optimizer = Adam::new();
    optimizer.set_hyperparams(hashmap! {
        "learning_rate" => 0.05,
        "decay" => 5e-5
    });

    let mut data_loss;
    let mut acc;
    let mut reg_loss;
    let mut loss;

    for _epoch in 0..10001 {
        dense1.forward(&x);
        activation1.forward(&dense1.outputs);
        dropout1.forward(activation1.outputs());
        dense2.forward(dropout1.outputs());
        data_loss = loss_activation.forward(&dense2.outputs, Arrayusize::Array1(&y));
        reg_loss = loss_activation.fn_loss.regularization_loss(
            &dense1) + loss_activation.fn_loss.regularization_loss(&dense2
        );
        loss = &data_loss + reg_loss;
        acc = accuracy(loss_activation.outputs(), &y);

        if _epoch % 100 == 0 {
            println!("epoch: {}", _epoch);
            println!("acc: {}", acc);
            print!("loss: {}\n", loss);
            println!("data_loss: {}", data_loss);
            println!("reg_loss: {}", reg_loss);
            println!("lr: {}\n", optimizer.current_learning_rate);
        }

        loss_activation.backward(&loss_activation.outputs().clone(), Arrayusize::Array1(&y));
        dense2.backward(loss_activation.dinputs());
        dropout1.backward(dense2.dinputs());
        activation1.backward(dropout1.dinputs());
        dense1.backward(activation1.dinputs());

        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.post_update_params();
    }

    let (x_val, y_val) = spiral_data(100, 3);

    dense1.forward(&x_val);
    activation1.forward(&dense1.outputs);
    dense2.forward(activation1.outputs());
    let val_loss = loss_activation.forward(
        &dense2.outputs, Arrayusize::Array1(&y_val)
    );

    let val_acc = accuracy(loss_activation.outputs(), &y_val);

    println!("Validation accuracy: {}", val_acc);
    println!("Validation loss: {}", val_loss);
}
