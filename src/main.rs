use ndarray::{self, Array1};
mod neural;
use neural::{Activation, Layer};

use crate::neural::import_network;
mod renderer;

fn main() {
    let grid = renderer::new_shared_grid();
    let grid_for_update = grid.clone();

    renderer::init(grid, neural::NeuralData::init(import_network("network1.txt")), move || {});
}

fn start() {
    println!("start")
}

fn update() {
    println!("update")
}

fn run_network() {
    let dataset = neural::get_data(1000000000);

    let mut network: Vec<Layer> = Vec::new();
    network.push(Layer::new(128, 784, Activation::ReLU));
    network.push(Layer::new(64, 128, Activation::ReLU));
    network.push(Layer::new(64, 64, Activation::ReLU));
    network.push(Layer::new(10, 64, Activation::ReLU));

    network = neural::import_network("network.txt");

    let mut total = 0.0;
    let mut wrong = 0.0;

    let learning_rate = 0.005;

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

        if *label as usize != max_idx {
            println!("Network predicted: {} Expected: {} Loss: {}", max_idx, label, neural::get_loss(&current_signals, &expected));
            neural::print_case(*label, pixels);
            wrong += 1.0;
        }
        total += 1.0;
        //for i in 0..10 {
        //    println!("{}", current_signals[i]);
        //}

        let mut output_error: Array1<f32> = &current_signals - &expected;

        for layer in &mut network.iter_mut().rev() {
            output_error = layer.backward(output_error, learning_rate)
        }
    }
    neural::export_network(&network, "network.txt").expect("Unable to save network");

    println!("Accuracy: {}", (total - wrong) / total);
}
