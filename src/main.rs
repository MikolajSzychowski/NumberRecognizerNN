use ndarray::{self, Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let dataset = get_data();

    let mut network: Vec<Layer> = Vec::new();
    network.push(Layer::new(128, 784, Activation::ReLU));
    network.push(Layer::new(64, 128, Activation::ReLU));
    network.push(Layer::new(10, 64, Activation::Sigmoid));

    for (label, pixels) in dataset.iter() {
        let mut current_signals = pixels.clone();

        for layer in &mut network {
            current_signals = layer.forward(&current_signals);
        }
        let mut max_idx = 0;
        let mut max_val = current_signals[0];
        for (idx, &val) in current_signals.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        let mut expected: Array1<f32> = Array1::zeros(10);
        expected[*label as usize] = 1.0;
        println!("Network predicted: {} Expected: {} Loss: {}", max_idx, label, get_loss(&current_signals, expected));
    }
}

fn get_loss(output: &Array1<f32>, expected: Array1<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..output.len() {
        sum += (output[i] - expected[i]).abs();
    }
    sum
}

struct Layer {
    biases: Array1<f32>,
    neurons: Array1<f32>,
    connections: Array2<f32>,
    activation: Activation,
}

impl Layer {
    fn new(neurons: usize, previous_size: usize, activation: Activation) -> Self {
        let std_dev = (2.0 / previous_size as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        Self {
            biases: Array1::zeros(neurons),
            neurons: Array1::zeros(neurons),
            connections: Array2::random((neurons, previous_size), normal_dist),
            activation,
        }
    }

    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut output = self.connections.dot(inputs);
        output += &self.biases;

        let act = self.activation;

        output.mapv_inplace(|x| act.apply(x));

        self.neurons = output.clone();
        output
    }
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Linear,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Linear => x,
        }
    }
}

#[allow(warnings)]
fn ReLU(x: f32) -> f32 {
    x.max(0.0)
}

#[allow(warnings)]
fn print_data(dataset: &Vec<Vec<u8>>) {
    for num in dataset {
        println!("\nDigit: {}", num[0]);

        for i in 1..785 {
            print!("{:3} ", num[i]);

            if i % 28 == 0 {
                println!();
            }
        }
    }
}

fn get_data() -> Vec<(u8, Array1<f32>)> {
    let file = File::open("data/mnist_train.csv").expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut nums = 0;
    let limit = 10;

    for line_result in reader.lines() {
        let line = line_result.expect("Failed to read line");

        if line.trim().is_empty() {
            continue;
        }
        if nums >= limit {
            break;
        }
        nums += 1;

        let mut tokens = line.split(',').map(|s| s.trim()).filter(|s| !s.is_empty());

        let label_str = tokens.next().expect("Missing label column");
        let label = label_str.parse::<u8>().expect("Failed to parse label");

        let pixels: Array1<f32> = tokens.map(|s| s.parse::<f32>().expect("Failed to parse pixel") / 255.0).collect();

        //let image_matrix = Array2::from_shape_vec((28, 28), pixels).expect("Data size mismatch: could not form a 28x28 matrix");

        out.push((label, pixels));
    }

    out
}
