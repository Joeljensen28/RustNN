#![allow(dead_code)]

use ndarray::{Array, Array1, Array2, Axis, Zip};
use ndarray_linalg::InnerProduct;

use crate::{activations::Softmax, utils::{clip, to_one_hot, to_sparse}};

pub struct CategoricalCrossEntropy {
    pub dinputs: Option<Array2<f64>>
}

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        CategoricalCrossEntropy{ dinputs: None }
    }

    pub fn forward_sparse(&self, y_pred: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        let y_pred_clipped = clip(&y_pred, f64::MIN, 1.0 - f64::MIN);

        let confs: Array1<f64> = y_pred_clipped
            .axis_iter(Axis(0))
            .zip(y_true)
            .map(|(sm, ct)| sm[*ct])
            .collect();

        let losses = confs.mapv(|x| -x.ln());
        losses.mean().expect("Losses array was unexpectedly empty")
    }

    pub fn forward_one_hot(&self, y_pred: &Array2<f64>, y_true: &Array2<usize>) -> f64 {
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
        losses.mean().expect("Losses array was unexpectedly empty")
    }

    pub fn backward_one_hot(&mut self, dvalues: Array2<f64>, y_true: &Array2<usize>) {
        let samples = dvalues.dim().0 as f64;
        let y_true_f64 = y_true.mapv(|x| x as f64);
        self.dinputs = Some((-y_true_f64 / dvalues) / samples);
    }

    pub fn backward_sparse(&mut self, dvalues: Array2<f64>, y_true: Array1<usize>) {
        let one_hot = to_one_hot(y_true, dvalues.dim().1);
        self.backward_one_hot(dvalues, &one_hot);
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

    pub fn forward_one_hot(&mut self, inputs: Array2<f64>, y_true: &Array2<usize>) -> f64 {
        self.fn_activation.forward(&inputs);
        self.output = Some(self.fn_activation.outputs().clone());
        self.fn_loss.forward_one_hot(&self.output.as_ref().expect("Outputs array was unexpectedly empty."), &y_true)
    }

    pub fn forward_sparse(&mut self, inputs: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        self.fn_activation.forward(&inputs);
        self.output = Some(self.fn_activation.outputs().clone());
        self.fn_loss.forward_sparse(&self.output.as_ref().expect("Outputs array was unexpectedly empty."), &y_true)
    }

    pub fn backward_sparse(&mut self, dvalues: &Array2<f64>, y_true: &Array1<usize>) {
        let samples = dvalues.dim().0 as f64;
        let mut dinputs = dvalues.clone();
        Zip::from(dinputs.rows_mut())
            .and(y_true)
            .for_each(|mut row, &col_idx| row[col_idx] -= 1.0);

        self.dinputs = Some(dinputs / samples);
    }

    pub fn backward_one_hot(&mut self, dvalues: &Array2<f64>, y_true: &Array2<usize>) {
        let y_true_sparse = to_sparse(y_true);
        self.backward_sparse(dvalues, &y_true_sparse);
    }

    pub fn outputs(&self) -> &Array2<f64> {
        self.output.as_ref().expect("Outputs unexpectedy empty. Be sure to call `forward` first.")
    }

    pub fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("Dinputs unexpectedy empty. Be sure to call `backward` first.")
    }
}