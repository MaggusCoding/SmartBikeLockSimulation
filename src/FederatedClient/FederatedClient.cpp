#include "FederatedClient/FederatedClient.h"

FederatedClient::FederatedClient(
    const std::vector<size_t>& topology,
    std::shared_ptr<DataPreprocessor> preprocessor,
    uint32_t seed)
    : network(topology, seed),
      preprocessor(preprocessor),
      rng(seed) {
}

void FederatedClient::train_on_sample(const std::vector<float>& features,
                                    const std::vector<float>& target,
                                    float learning_rate) {
    network.train(features, target, learning_rate);
}

void FederatedClient::train_on_random_sample(float learning_rate) {
    // Get a single random training sample
    auto batch = preprocessor->get_training_batch(1);
    if (!batch.empty()) {
        const auto& sample = batch[0];
        
        
        // Perform training
        network.train(sample.features, sample.target, learning_rate);
        
        // Get weights after training
        auto weights_after = network.get_flat_weights();
    }
}

std::vector<float> FederatedClient::get_weights() const {
    return network.get_flat_weights();
}

void FederatedClient::set_weights(const std::vector<float>& weights) {
    network.set_flat_weights(weights);
}

std::vector<float> FederatedClient::predict(const std::vector<float>& features) {
    return network.forward(features);
}