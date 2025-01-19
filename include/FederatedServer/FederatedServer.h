#ifndef FEDERATED_SERVER_H
#define FEDERATED_SERVER_H

#include <vector>
#include <memory>

class FederatedServer {
public:
    // FedAvg implementation
    std::vector<float> average_weights(const std::vector<std::vector<float>>& client_weights);
    
private:
    // Helper method to verify weights are compatible
    bool verify_weights(const std::vector<std::vector<float>>& client_weights) const;
};

#endif