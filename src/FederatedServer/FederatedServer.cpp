#include "FederatedServer/FederatedServer.h"
#include <stdexcept>
#include <algorithm>
#include <random>


FederatedServer::FederatedServer(uint32_t seed) : rng(seed) {}


std::vector<size_t> FederatedServer::select_clients(size_t total_clients, float client_fraction) {
    if (client_fraction <= 0.0f || client_fraction > 1.0f) {
        throw std::runtime_error("Client fraction must be between 0 and 1");
    }
    
    // Calculate number of clients to select
    size_t num_selected = std::max(
        size_t(1), 
        static_cast<size_t>(total_clients * client_fraction)
    );
    
    // Create vector of all client indices
    std::vector<size_t> all_clients(total_clients);
    std::iota(all_clients.begin(), all_clients.end(), 0);
    
    // Shuffle and select first num_selected clients
    std::shuffle(all_clients.begin(), all_clients.end(), rng);
    
    return std::vector<size_t>(
        all_clients.begin(), 
        all_clients.begin() + num_selected
    );
}

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