#ifndef FEDERATED_SIMULATION_H
#define FEDERATED_SIMULATION_H

#include "FederatedServer.h"
#include "DataLoader/DataLoader.h"
#include <random>

class FederatedSimulation {
public:
    FederatedSimulation(const std::string& data_path, 
                       const std::vector<size_t>& topology,
                       size_t num_clients);
    
    // Simulation control
    void run_simulation(int rounds, float learning_rate);
    
    // Dataset distribution
    void distribute_data_to_clients();
    
private:
    std::unique_ptr<FederatedServer> server;
    std::vector<std::shared_ptr<FederatedClient>> clients;
    std::vector<MotionSample> full_dataset;
    std::mt19937 rng;
    
    // Helper methods
    std::vector<std::vector<MotionSample>> split_dataset(size_t num_clients);
};

#endif