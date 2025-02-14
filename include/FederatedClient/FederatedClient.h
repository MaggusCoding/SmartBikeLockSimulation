#ifndef FEDERATED_CLIENT_H
#define FEDERATED_CLIENT_H

#include "NeuralNetwork/NeuralNetwork.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include <memory>
#include <random>

class FederatedClient {
public:
    // Initialize with network topology and preprocessor
    FederatedClient(const std::vector<size_t>& topology, std::shared_ptr<DataPreprocessor> preprocessor, uint32_t seed);
    
    // Core FL operations
    void train_on_sample(const std::vector<float>& features, 
                        const std::vector<float>& target,
                        float learning_rate);
    std::vector<float> get_weights() const;
    void set_weights(const std::vector<float>& weights);
    
    // Inference
    std::vector<float> predict(const std::vector<float>& features);
    
    // Access to neural network for evaluation
    const NeuralNetwork& get_network() const { return network; }
    NeuralNetwork& get_network() { return network; }

private:
    NeuralNetwork network;
    std::shared_ptr<DataPreprocessor> preprocessor;
    std::mt19937 rng;
};

#endif