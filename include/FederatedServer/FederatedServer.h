#ifndef FEDERATED_SERVER_H
#define FEDERATED_SERVER_H

#include <vector>
#include <memory>
#include "FederatedClient.h"

class FederatedServer {
public:
    FederatedServer(const std::vector<size_t>& topology);
    
    // Client management
    void register_client(std::shared_ptr<FederatedClient> client);
    
    // Federated learning process
    void aggregate_weights();
    void broadcast_weights();
    
    // Training coordination
    void train_round(float learning_rate);
    
private:
    std::vector<std::shared_ptr<FederatedClient>> clients;
    std::vector<float> global_weights;
    std::vector<size_t> topology;
};
#endif