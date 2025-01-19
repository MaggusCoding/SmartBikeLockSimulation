#include "FederatedServer/FederatedServer.h"
#include <stdexcept>

std::vector<float> FederatedServer::average_weights(
    const std::vector<std::vector<float>>& client_weights) {
    
    if (!verify_weights(client_weights)) {
        throw std::runtime_error("Invalid client weights for averaging");
    }
    
    // Get dimensions
    const size_t num_clients = client_weights.size();
    const size_t num_weights = client_weights[0].size();
    
    // Initialize result vector
    std::vector<float> averaged_weights(num_weights, 0.0f);
    
    // Simple averaging (equal weight for each client)
    for (size_t i = 0; i < num_weights; i++) {
        float sum = 0.0f;
        for (size_t client = 0; client < num_clients; client++) {
            sum += client_weights[client][i];
        }
        averaged_weights[i] = sum / num_clients;
    }
    
    return averaged_weights;
}

bool FederatedServer::verify_weights(
    const std::vector<std::vector<float>>& client_weights) const {
    
    if (client_weights.empty()) {
        return false;
    }
    
    const size_t expected_size = client_weights[0].size();
    for (const auto& weights : client_weights) {
        if (weights.size() != expected_size) {
            return false;
        }
    }
    
    return true;
}