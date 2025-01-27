#ifndef FEDERATED_SERVER_H
#define FEDERATED_SERVER_H

#include <vector>
#include <memory>
#include <random>

class FederatedServer {
public:
    explicit FederatedServer(uint32_t seed = 42);
    // FedAvg implementation
    std::vector<float> average_weights(const std::vector<std::vector<float>>& client_weights);
    std::vector<size_t> select_clients(size_t total_clients, float client_fraction);
    
private:
    // Helper method to verify weights are compatible
    bool verify_weights(const std::vector<std::vector<float>>& client_weights) const;
    std::mt19937 rng; // RNG for client selection
};

#endif