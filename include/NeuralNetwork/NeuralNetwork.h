#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <random>
#include <cmath>

class Layer {
public:
    Layer(size_t inputs, size_t outputs, uint32_t seed);

    std::vector<float> forward(const std::vector<float>& inputs);
    void backward(const std::vector<float>& inputs, std::vector<float>& gradients, float learning_rate);

    size_t input_size() const { return weights.size() > 0 ? weights[0].size() : 0; }
    size_t output_size() const { return weights.size(); }

    // Direct weight access for distributed learning
    const std::vector<std::vector<float>>& get_weights() const { return weights; }
    void set_weights(const std::vector<std::vector<float>>& new_weights) { weights = new_weights; }

    // Getter for last outputs
    const std::vector<float>& get_last_outputs() const { return last_outputs; }

private:
    std::vector<std::vector<float>> weights;  // [output_neurons][input_neurons]
    std::vector<float> biases;
    std::vector<float> last_outputs;  // Cache for backprop

    float activate(float x) const;
    float activate_derivative(float x) const;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<size_t>& topology, uint32_t seed);

    std::vector<float> forward(const std::vector<float>& inputs);
    void train(const std::vector<float>& inputs, const std::vector<float>& targets, float learning_rate);

    // Methods for distributed learning
    std::vector<float> get_flat_weights() const;
    void set_flat_weights(const std::vector<float>& weights);

private:
    std::vector<Layer> layers;
};

#endif