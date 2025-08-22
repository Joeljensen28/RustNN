#![allow(dead_code)]

use core::f64;

use ndarray::{Array2, Axis};

use crate::utils::diagflat;

pub struct ReLU {
    pub outputs: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { 
            outputs: None, 
            inputs: None,
            dinputs: None
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = Some(inputs.clone());
        self.outputs = Some(inputs.mapv(|v| v.max(0.0)));
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        let mut dinputs = dvalues.clone();
        dinputs.zip_mut_with(
            self.outputs.as_ref().expect("No output set. Make sure to call `forward` before `backward`."), 
            |d, &zv| if zv <= 0.0 { *d = 0.0 }
        );
        self.dinputs = Some(dinputs);
    }

    pub fn outputs(&self) -> &Array2<f64> {
        self.outputs.as_ref().expect("No output set. Make sure to call `forward` first.")
    }

    pub fn dinputs(&self) -> &Array2<f64> {
        self.dinputs.as_ref().expect("No dinputs set. Make sure to call `backward` first.")
    }
}

pub struct Softmax {
    pub inputs: Option<Array2<f64>>,
    pub outputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>
}

impl Softmax {
    pub fn new() -> Self {
        Softmax{
            inputs: None,
            outputs: None,
            dinputs: None
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = Some(inputs.clone());
        let sample_maxes = inputs.map_axis(
            Axis(1), |r| r.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        ).insert_axis(Axis(1));
        let input_norm = inputs - sample_maxes;
        let exp_values = input_norm.mapv(f64::exp);
        let sample_sum = exp_values.sum_axis(Axis(1)).insert_axis(Axis(1));
        let probs = &exp_values / &sample_sum;
        self.outputs = Some(probs);
    }

    pub fn backward(&mut self, dvalues: Array2<f64>) {
        let mut dinputs: Array2<f64> = Array2::zeros(dvalues.raw_dim());

        for 
            (index, (single_output, single_dvalues)) 
            in self.outputs
                .as_ref()
                .expect("No output set. Make sure to call `forward` first.")
                .rows()
                .into_iter()
                .zip(dvalues.rows())
                .enumerate() {
                    let output_col = &single_output
                        .into_shape((single_output.len(), 1))
                        .unwrap()
                        .to_owned();

                    let jacobian_matrix = diagflat(output_col) - output_col.dot(&output_col.t());

                    dinputs.row_mut(index).assign(&jacobian_matrix.dot(&single_dvalues));
        }

        self.dinputs = Some(dinputs);
    }

    pub fn outputs(&self) -> &Array2<f64> {
        self.outputs.as_ref().expect("No output set. Make sure to call `forward` first.")
    }
}