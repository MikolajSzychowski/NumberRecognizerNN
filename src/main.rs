use ndarray::{self, ArcArray, Array1, Array2};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let dataset = get_data();

    let mut network: Vec<Layer> = vec![Layer::new(128, 784), Layer::new(64, 128), Layer::new(10, 64)];

    for (label, pixels) in dataset.iter().take(10) {
        let mut current_signals = pixels.view().to_shape(784).unwrap().to_owned();

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
        println!("Network predicted: {} Expected: {}", max_idx, label);
    }
}

struct Layer {
    biases: Array1<f32>,
    neurons: Array1<f32>,
    connections: Array2<f32>,
}

impl Layer {
    fn new(neurons: usize, previous_size: usize) -> Self {
        let std_dev = (2.0 / previous_size as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        Self {
            biases: Array1::zeros(neurons),
            neurons: Array1::zeros(neurons),
            connections: Array2::random((neurons, previous_size), normal_dist),
        }
    }

    fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        let mut output = self.connections.dot(inputs);
        output += &self.biases;

        output.mapv_inplace(|x| {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        });
        self.neurons = output.clone();
        output
    }
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

fn get_data() -> Vec<(u8, Array2<f32>)> {
    let file = File::open("data/mnist_train.csv").expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut nums = 0;
    let limit = 100;

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

        let pixels: Vec<f32> = tokens.map(|s| s.parse::<f32>().expect("Failed to parse pixel") / 255.0).collect();

        let image_matrix = Array2::from_shape_vec((28, 28), pixels).expect("Data size mismatch: could not form a 28x28 matrix");

        out.push((label, image_matrix));
    }

    out
}
