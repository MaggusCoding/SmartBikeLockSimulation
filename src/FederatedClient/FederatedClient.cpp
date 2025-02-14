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


std::vector<float> FederatedClient::get_weights() const {
    return network.get_flat_weights();
}

void FederatedClient::set_weights(const std::vector<float>& weights) {
    network.set_flat_weights(weights);
}

std::vector<float> FederatedClient::predict(const std::vector<float>& features) {
    return network.forward(features);
}