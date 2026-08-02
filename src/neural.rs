use ndarray::{self, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};

pub struct NeuralData {
    pub network: Vec<Layer>,
}

impl NeuralData {
    pub fn init(import_network: Vec<Layer>) -> Self {
        NeuralData {
            network: import_network,
        }
    }

    pub fn get_network(&self) -> &Vec<Layer> {
        &self.network
    }

    pub fn calculate_network(&mut self, pixels: &Array1<f32>) -> i32 {
        let mut current_signals = pixels.clone();

        for layer in &mut self.network {
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
        println!("Network predicted: {}", max_idx);
        max_idx as i32
    }
}

pub struct Layer {
    pub biases: Array1<f32>,
    pub neurons: Array1<f32>,
    pub inputs: Array1<f32>,
    pub connections: Array2<f32>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(neurons: usize, previous_size: usize, activation: Activation) -> Self {
        let std_dev = (2.0 / previous_size as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        Self {
            biases: Array1::zeros(neurons),
            neurons: Array1::zeros(neurons),
            connections: Array2::random((neurons, previous_size), normal_dist),
            activation,
            inputs: Array1::zeros(neurons),
        }
    }

    pub fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        self.inputs = inputs.clone();
        let mut output = self.connections.dot(inputs);
        output += &self.biases;

        let act = self.activation;

        output.mapv_inplace(|x| act.apply(x));

        self.neurons = output.clone();
        output
    }

    pub fn backward(&mut self, output_error: Array1<f32>, learning_rate: f32) -> Array1<f32> {
        let activation_function = self.activation;
        let derivative = self.neurons.mapv(|x| activation_function.derivative(x));

        let local_error = output_error * &derivative;
        let input_error = self.connections.t().dot(&local_error);

        let local_error_col = local_error.view().insert_axis(Axis(1));
        let inputs_row = self.inputs.view().insert_axis(Axis(0));
        let weight_gradients = local_error_col.dot(&inputs_row);

        self.connections -= &(&weight_gradients * learning_rate);
        self.biases -= &(&local_error * learning_rate);

        input_error
    }
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Linear,
}

impl Activation {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Linear => x,
        }
    }
    pub fn derivative(self, output_val: f32) -> f32 {
        match self {
            Activation::ReLU => {
                if output_val > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => output_val * (1.0 - output_val),
            Activation::Linear => 1.0,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Activation::ReLU => "ReLU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Linear => "Linear",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "ReLU" => Activation::ReLU,
            "Sigmoid" => Activation::Sigmoid,
            "Linear" => Activation::Linear,
            _ => panic!("Invalid activation function: {}", s),
        }
    }
}

pub fn export_network(network: &Vec<Layer>, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    writeln!(file, "{}", network.len())?;

    for layer in network {
        writeln!(file, "{} {} {}", layer.biases.len(), layer.inputs.len(), layer.activation.as_str())?;

        for bias in &layer.biases {
            write!(file, "{} ", bias)?;
        }
        writeln!(file)?;

        for weight in &layer.connections {
            write!(file, "{} ", weight)?;
        }
        writeln!(file)?;
    }

    println!("Saved network to file: {}", filename);
    Ok(())
}

pub fn import_network(filename: &str) -> Vec<Layer> {
    let content = std::fs::read_to_string(filename).expect("Nie udało się odczytać pliku");
    let mut tokens = content.split_whitespace();

    let num_layers: usize = tokens.next().expect("Pusty plik").parse().expect("Błąd parsowania liczby warstw");
    let mut network = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        let num_neurons: usize = tokens.next().unwrap().parse().unwrap();
        let num_inputs: usize = tokens.next().unwrap().parse().unwrap();

        let activation_str = tokens.next().unwrap();
        let activation = Activation::from_str(activation_str);

        let mut biases_vec = Vec::with_capacity(num_neurons);
        for _ in 0..num_neurons {
            let b: f32 = tokens.next().unwrap().parse().unwrap();
            biases_vec.push(b);
        }
        let biases = Array1::from(biases_vec);

        let total_weights = num_neurons * num_inputs;
        let mut weights_vec = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            let w: f32 = tokens.next().unwrap().parse().unwrap();
            weights_vec.push(w);
        }

        let connections = Array2::from_shape_vec((num_neurons, num_inputs), weights_vec).expect("Błąd podczas odtwarzania macierzy wag");

        network.push(Layer {
            biases,
            connections,
            activation,
            neurons: Array1::zeros(num_neurons),
            inputs: Array1::zeros(num_inputs),
        });
    }

    network
}

pub fn get_data(limit: u32) -> Vec<(u8, Array1<f32>)> {
    let file = File::open("data/mnist_train.csv").expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut nums = 0;

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

        out.push((label, pixels));
    }

    out
}

pub fn get_loss(output: &Array1<f32>, expected: &Array1<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..output.len() {
        sum += (output[i] - expected[i]).abs().powi(2);
    }
    sum
}

pub fn print_case(label: u8, pixels: &Array1<f32>) {
    println!("\nDigit: {}", label);

    for (i, &pixel) in pixels.iter().enumerate() {
        let val = (pixel * 255.0) as u8;
        print!("{:3} ", val);

        if (i + 1) % 28 == 0 {
            println!();
        }
    }
}
