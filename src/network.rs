//! impl of neural network

use rand::{rngs::ThreadRng, Rng};

use crate::{
    common::{Float, IMG_RESOLUTION},
    idx::{ImageSet, LabelSet},
};

///
/// learning rate
///
const ALPHA: Float = 0.1;
const N_HIDDEN: usize = 15;
const N_OUTPUT: usize = 10;

///
/// http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits
/// a neural network consist of 3 layers
/// input layer: just every pixels in img
/// one only hidden layer
/// output layer: 10 neural output 10 possibility of 10 digits
///
pub struct Network {
    layers: [Layer; 2],
}

impl Network {
    pub fn new() -> Network {
        let mut rng = rand::thread_rng();

        let mut hidden_neurons = Vec::with_capacity(N_HIDDEN);
        for _ in 0..N_HIDDEN {
            hidden_neurons.push(Neuron::new(IMG_RESOLUTION, &mut rng));
        }
        let hidden_layer = Layer {
            neurons: hidden_neurons,
        };

        let mut output_neurons = Vec::with_capacity(N_OUTPUT);
        for _ in 0..N_OUTPUT {
            output_neurons.push(Neuron::new(N_HIDDEN, &mut rng));
        }
        let output_layer = Layer {
            neurons: output_neurons,
        };

        Network {
            layers: [hidden_layer, output_layer],
        }
    }

    pub fn train(&mut self, imgs: &ImageSet, labels: &LabelSet, round: usize) {
        let alpha = ALPHA / round as Float;
        // outputs of each layer, `a` in every post
        let mut a = vec![vec![0.0; N_HIDDEN], vec![0.0; N_OUTPUT]];
        // deltas, `dz` in every post, `dw` & `db` are ignored
        let mut dz = vec![vec![0.0; N_HIDDEN], vec![0.0; N_OUTPUT]];

        for r in 0..round {
            let mut err = 0;
            for (img, actual) in imgs.iter().zip(labels.iter()) {
                // 1. feed forward
                let inputs = img.all_pixels();
                // input => hidden
                for (i, neuron) in self.layers[0].neurons.iter().enumerate() {
                    // input layer
                    a[0][i] = sigmoid(neuron.z(inputs));
                }
                // hidden => ouput
                for (i, neuron) in self.layers[1].neurons.iter().enumerate() {
                    // full conn
                    a[1][i] = sigmoid(neuron.z(&a[0]));
                }

                // 2. count final err
                let last_outputs = a.last().unwrap();
                let pred = the_num_is(&last_outputs);
                if pred != actual {
                    err += 1;
                }

                // 3. back propagation => A MESS
                // delta in output layer
                debug_assert_eq!(dz[1].len(), a[1].len());
                for (i, (d, a)) in dz[1].iter_mut().zip(a[1].iter()).enumerate() {
                    // output layer err = a - actual
                    // autual 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    let err = if i == actual as usize { a - 1.0 } else { *a };
                    *d = err * sigmoid_derivative(*a);
                }
                // // delta in hidden layer
                debug_assert_eq!(dz[0].len(), a[0].len());
                for i in 0..N_HIDDEN {
                    // hidden layer err = output deltas * output weights
                    let mut err = 0.0;
                    for j in 0..N_OUTPUT {
                        err += dz[1][j] * self.layers[1].neurons[j].weights[i];
                    }

                    dz[0][i] = err * sigmoid_derivative(a[0][i]);
                }

                // 4. update params
                for (i, layer) in self.layers.iter_mut().enumerate() {
                    for (j, neuron) in layer.neurons.iter_mut().enumerate() {
                        for (k, weight) in neuron.weights.iter_mut().enumerate() {
                            let dw = if i == 1 {
                                // output layer dw
                                a[0][k] * dz[1][j]
                            } else {
                                // hidden layer dw
                                inputs[k] * dz[0][j]
                            };
                            *weight -= alpha * dw;
                        }
                        neuron.bias -= alpha * dz[i][j];
                    }
                }
            }

            println!("round:{} err:{}", r, err);
        }
    }

    pub fn predict(&mut self, imgs: &ImageSet, labels: &LabelSet) {
        let mut total = 0;
        let mut err = 0;
        let mut a = vec![vec![0.0; N_HIDDEN], vec![0.0; N_OUTPUT]];

        for (img, l) in imgs.iter().zip(labels.iter()) {
            for (i, neuron) in self.layers[0].neurons.iter().enumerate() {
                // input layer
                a[0][i] = sigmoid(neuron.z(img.all_pixels()));
            }
            // hidden => ouput
            for (i, neuron) in self.layers[1].neurons.iter().enumerate() {
                // full conn
                a[1][i] = sigmoid(neuron.z(&a[0]));
            }
            let num = the_num_is(&a[1]);

            total += 1;
            if num != l {
                err += 1;
            }
        }

        println!(
            "total: {} err: {} precision: {}%",
            total,
            err,
            ((total - err) as f32 / total as f32) * 100.0
        );
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
struct Neuron {
    bias: Float,
    weights: Vec<Float>,
}

impl Neuron {
    fn new(n: usize, rng: &mut ThreadRng) -> Neuron {
        let mut weights = vec![0.0; n];
        for w in weights.iter_mut() {
            *w = rng.gen();
        }
        Neuron { bias: 0.0, weights }
    }

    fn z(&self, inputs: &[Float]) -> Float {
        debug_assert_eq!(self.weights.len(), inputs.len());
        // dot multiply
        let mut wx = 0.0;
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            wx += w * x;
        }

        wx + self.bias
    }
}

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: Float) -> Float {
    sigmoid(x) * (1.0 - sigmoid(x))
}

///
/// 10 outputs neurons mapping to 10 digits
/// max possibility wins
/// which means: max idx [0-9] is my guess
///
fn the_num_is(outputs: &[f32]) -> u8 {
    // println!("the_num_is:{:?}", outputs);
    debug_assert_eq!(outputs.len(), 10);
    let mut max_possible = 0.0;
    let mut max_idx = 0;
    for (i, p) in outputs.iter().enumerate() {
        if p > &max_possible {
            max_idx = i as u8;
            max_possible = *p;
        }
    }

    max_idx
}
