#include "NeuralNetwork/NeuralNetwork.h"

Layer::Layer(size_t inputs, size_t outputs, uint32_t seed = 42) : 
    weights(outputs, std::vector<float>(inputs)),
    last_outputs(outputs) {
    
    // Initialize with Xavier/Glorot initialization
    std::mt19937 gen(seed);
    float weight_range = std::sqrt(6.0f / (inputs + outputs));
    std::uniform_real_distribution<float> d(-weight_range, weight_range);
    
    for(auto& neuron_weights : weights) {
        for(float& weight : neuron_weights) {
            weight = d(gen);
        }
    }
}

float Layer::activate(float x) const {
    // Sigmoid activation
    return 1.0f / (1.0f + std::exp(-x));
}

float Layer::activate_derivative(float x) const {
    // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    return x * (1.0f - x);
}

std::vector<float> Layer::forward(const std::vector<float>& inputs) {
    last_outputs.resize(weights.size());
    
    for(size_t i = 0; i < weights.size(); i++) {
        float sum = 0.0f;  // No bias
        for(size_t j = 0; j < weights[i].size(); j++) {
            sum += weights[i][j] * inputs[j];
        }
        last_outputs[i] = activate(sum);
    }
    
    return last_outputs;
}

void Layer::backward(const std::vector<float>& inputs, 
                    std::vector<float>& gradients, 
                    float learning_rate) {
    std::vector<float> next_gradients(inputs.size(), 0.0f);
    
    for(size_t i = 0; i < weights.size(); i++) {
        float delta = gradients[i] * activate_derivative(last_outputs[i]);
        
        // Update weights
        for(size_t j = 0; j < weights[i].size(); j++) {
            next_gradients[j] += weights[i][j] * delta;
            weights[i][j] -= learning_rate * delta * inputs[j];
        }
    }
    
    gradients = next_gradients;
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, uint32_t seed = 42) {
    for(size_t i = 0; i < topology.size() - 1; i++) {
        // Use seed + i to get different but deterministic initialization per layer
        layers.emplace_back(topology[i], topology[i + 1], seed + i);
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& inputs) {
    std::vector<float> current = inputs;
    for(auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

void NeuralNetwork::train(const std::vector<float>& inputs, 
                         const std::vector<float>& targets, 
                         float learning_rate) {
    // Forward pass
    auto outputs = forward(inputs);
    
    // Calculate output layer gradients
    std::vector<float> gradients = outputs;
    for(size_t i = 0; i < gradients.size(); i++) {
        gradients[i] = outputs[i] - targets[i];
    }
    
    // Backward pass
    std::vector<float> current_inputs = inputs;
    for(int i = layers.size() - 1; i >= 0; i--) {
        layers[i].backward(i == 0 ? inputs : layers[i-1].get_last_outputs(), 
                         gradients, learning_rate);
    }
}

std::vector<float> NeuralNetwork::get_flat_weights() const {
    std::vector<float> flat_weights;
    for(const auto& layer : layers) {
        const auto& weights = layer.get_weights();
        for(const auto& neuron : weights) {
            flat_weights.insert(flat_weights.end(), neuron.begin(), neuron.end());
        }
    }
    return flat_weights;
}

void NeuralNetwork::set_flat_weights(const std::vector<float>& weights) {
    size_t offset = 0;
    for(auto& layer : layers) {
        size_t inputs = layer.input_size();
        size_t outputs = layer.output_size();
        size_t layer_weights = inputs * outputs;
        
        std::vector<std::vector<float>> layer_weight_matrix(outputs, 
                                                          std::vector<float>(inputs));
        for(size_t i = 0; i < outputs; i++) {
            for(size_t j = 0; j < inputs; j++) {
                layer_weight_matrix[i][j] = weights[offset + i * inputs + j];
            }
        }
        
        layer.set_weights(layer_weight_matrix);
        offset += layer_weights;
    }
}