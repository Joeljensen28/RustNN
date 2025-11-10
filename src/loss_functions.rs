#![allow(dead_code)]

use ndarray::{Array, Array1, Array2, Axis, Zip};
use ndarray_linalg::InnerProduct;

use crate::{activations::Softmax, utils::{clip, to_one_hot, to_sparse, Arrayusize, log}, layers::Dense};

// TODO: Use traits and enums to clean up this mess. Especially use enums for the forward functions so there doesn't have 
// to be a "forward_sparse" and "forward_one_hot"....

pub trait Loss {
    fn regularization_loss(&self, layer: &Dense) -> f64 {
        let mut regularization_loss = 0.0;
        if layer.weight_regularizer_l1 > 0.0 {
            regularization_loss += &(layer.weight_regularizer_l1 * layer.weights.mapv(|x| x.abs()).sum());
        }

        if layer.weight_regularizer_l2 > 0.0 {
            regularization_loss += &(layer.weight_regularizer_l2 * (&layer.weights * &layer.weights).sum());
        }

        if layer.bias_regularizer_l1 > 0.0 {
            regularization_loss += &(layer.bias_regularizer_l1 * layer.biases.mapv(|x| x.abs()).sum());
        }

        if layer.bias_regularizer_l2 > 0.0 {
            regularization_loss += &(layer.bias_regularizer_l2 * (&layer.biases * &layer.biases).sum());
        }

        regularization_loss
    }

    fn calculate(&self, output: &Array2<f64>, y: Arrayusize) -> f64 {
        let sample_losses = self.forward(output, y);
        let data_loss = sample_losses.mean().expect("sample_losses.mean() failed. Is the array empty?");
        data_loss
    }

    fn forward(&self, y_pred: &Array2<f64>, y_true: Arrayusize) -> Array1<f64>;

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: Arrayusize);

    fn dinputs(&self) -> &Array2<f64>;

    fn new() -> Self;
}

pub struct CategoricalCrossEntropy {
    pub dinputs: Option<Array2<f64>>
}

impl Loss for CategoricalCrossEntropy {
    fn new() -> Self {
        CategoricalCrossEntropy{ dinputs: None }
    }

    fn forward(&self, y_pred: &Array2<f64>, y_true: Arrayusize) -> Array1<f64> {
        match y_true {
            Arrayusize::Array1(y_true) => {
                let y_pred_clipped = clip(&y_pred, f64::MIN, 1.0 - f64::MIN);

                let confs: Array1<f64> = y_pred_clipped
                    .axis_iter(Axis(0))
                    .zip(y_true)
                    .map(|(sm, ct)| sm[*ct as usize])
                    .collect();

                let losses = confs.mapv(|x| -x.ln());
                losses
            }

            Arrayusize::Array2(y_true) => {
                let y_pred_clipped = clip(&y_pred, f64::MIN, 1.0 - f64::MIN);

                let confs: Array1<f64> = y_pred_clipped
                    .axis_iter(Axis(0))
                    .zip(y_true.axis_iter(Axis(0)))
                    .map(|(sm, ct)| {
                        let class_idx = ct
                            .iter()
                            .position(|&x| x == 1)
                            .expect("No target class in label");
                        sm[class_idx]
                    })
                    .collect();

                let losses = confs.mapv(|x| -x.ln());
                losses
            }
        }
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: Arrayusize) {
        let y_true = match y_true {
            Arrayusize::Array1(inner) => {
                to_one_hot(inner, dvalues.dim().1)
            }
            Arrayusize::Array2(inner) => {
                inner.clone()
            }
        };
        let samples = dvalues.dim().0 as f64;
        let y_true_f64 = y_true.mapv(|x| x as f64);
        self.dinputs = Some((-y_true_f64 / dvalues) / samples);
    }

    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("dinputs not yet set. Make sure to call `.forward()` first.")
    }
}

pub struct SoftmaxCategoricalCrossEntropy {
    pub fn_activation: Softmax,
    pub fn_loss: CategoricalCrossEntropy,
    pub output: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>
}

impl SoftmaxCategoricalCrossEntropy {
    pub fn new() -> Self {
        let fn_activation = Softmax::new();
        let fn_loss = CategoricalCrossEntropy::new();

        SoftmaxCategoricalCrossEntropy{
            fn_activation: fn_activation,
            fn_loss: fn_loss,
            output: None,
            dinputs: None
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>, y_true: Arrayusize) -> f64 {
        match y_true {
            Arrayusize::Array1(inner) => {
                self.fn_activation.forward(&inputs);
                self.output = Some(self.fn_activation.outputs().clone());
                self.fn_loss.calculate(
                    &self.output.as_ref().expect("Outputs array was unexpectedly empty."), 
                    Arrayusize::Array1(&inner)
                )
            },
            Arrayusize::Array2(inner) => {
                self.fn_activation.forward(&inputs);
                self.output = Some(self.fn_activation.outputs().clone());
                self.fn_loss.calculate(
                    &self.output.as_ref().expect("Outputs array was unexpectedly empty."), 
                    Arrayusize::Array2(&inner)
                )
            }
        }
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>, y_true: Arrayusize) {
        let y_true = match y_true {
            Arrayusize::Array1(inner) => {
                inner.clone()
            }
            Arrayusize::Array2(inner) => {
                to_sparse(inner)
            }
        };
        let samples = dvalues.dim().0 as f64;
        let mut dinputs = dvalues.clone();
        Zip::from(dinputs.rows_mut())
            .and(&y_true)
            .for_each(|mut row, &col_idx| row[col_idx] -= 1.0);

        self.dinputs = Some(dinputs / samples);
    }

    pub fn outputs(&self) -> &Array2<f64> {
        self.output.as_ref().expect("Outputs unexpectedy empty. Be sure to call `forward` first.")
    }

    pub fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("Dinputs unexpectedy empty. Be sure to call `backward` first.")
    }
}

pub struct BinaryCrossEntropy {
    pub dinputs: Option<Array2<f64>>
}

impl Loss for BinaryCrossEntropy {
    fn new() -> Self {
        BinaryCrossEntropy { dinputs: None }
    }

    fn forward(&self, y_pred: &Array2<f64>, y_true: Arrayusize) -> Array1<f64> {
        let y_pred_clipped = clip(y_pred, 1e-7, 1.0 - 1e-7);

        let y_true = match y_true {
            Arrayusize::Array1(_inner) => {
                panic!("Cannot pass any <2D array) into BinaryCrossEntropy.")
            }
            Arrayusize::Array2(inner) => {
                &inner.mapv(|x| x as f64)
            }
        };

        let sample_losses = -(y_true * log(&y_pred_clipped) +
            (1.0 - y_true) * log(&(1.0 - y_pred_clipped)));
            
        let sample_losses = sample_losses
            .mean_axis(Axis(sample_losses.ndim() - 1))
            .expect("sample_losses unexpectedly empty and cannot calculate mean.");

        sample_losses
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: Arrayusize) {
        let samples = dvalues.len();

        let outputs = dvalues.dim().1;

        let clipped_dvalues = clip(dvalues, 1e-7, 1.0 - 1e-7);

        let y_true = match y_true {
            Arrayusize::Array1(_inner) => {
                panic!("Cannot pass any <2D array) into BinaryCrossEntropy.")
            }
            Arrayusize::Array2(inner) => {
                &inner.mapv(|x| x as f64)
            }
        };

        self.dinputs = Some(
            (-(y_true / &clipped_dvalues - (1.0 - y_true) / (1.0 - clipped_dvalues)) / (outputs as f64)) / (samples as f64)
        );
    }

    fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("dinputs not yet set. Make sure to call `.backward()` first.")
    }
}