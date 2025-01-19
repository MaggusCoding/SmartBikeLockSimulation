#ifndef FEDERATED_CLIENT_H
#define FEDERATED_CLIENT_H

#include "NeuralNetwork/NeuralNetwork.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include <memory>
#include <string>

class FederatedClient {
public:
    FederatedClient(const std::vector<size_t>& topology, 
                   const std::vector<MotionSample>& local_dataset,
                   const std::string& client_id);
    
    // Training methods
    void train_epoch(float learning_rate);
    
    // Model synchronization
    std::vector<float> get_weights() const;
    void set_weights(const std::vector<float>& weights);
    
    // Evaluation
    float evaluate() const;
    
private:
    std::string client_id;
    std::unique_ptr<NeuralNetwork> network;
    DataPreprocessor data_preprocessor;
};
#endif