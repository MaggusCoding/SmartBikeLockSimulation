/*#include "NeuralNetwork/NeuralNetwork.h"
#include "DataLoader/DataLoader.h"
#include "FeatureExtractor/FeatureExtractor.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include "Metrics/Metrics.h"
#include "FederatedClient/FederatedClient.h"
#include "FederatedServer/FederatedServer.h"
#include "HPO/HyperParameterOptimizer.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>

std::vector<OptimizationResult> HyperParameterOptimizer::run_optimization() {
    std::vector<OptimizationResult> results;
    size_t total_combinations = topologies.size() * samples_per_round.size() * 
                              learning_rates.size() * client_fractions.size();
    size_t current_combination = 0;
    
    // Create log file
    std::ofstream log_file("hyperparameter_optimization.csv");
    log_file << "Topology,SamplesPerRound,LearningRate,ClientFraction,RoundsToTarget,FinalAccuracy,ReachedTarget\n";
    
    for (const auto& topology : topologies) {
        for (size_t spr : samples_per_round) {
            for (float lr : learning_rates) {
                for (float cf : client_fractions) {
                    current_combination++;
                    std::cout << "\nTesting combination " << current_combination 
                              << "/" << total_combinations << "\n";
                    
                    HyperParameters params{spr, lr, topology, cf};
                    std::cout << "Parameters: " << params.to_string() << "\n";
                    
                    OptimizationResult result = evaluate_parameters(params);
                    results.push_back(result);
                    
                    // Log result
                    log_file << topology.to_string() << ","
                            << spr << ","
                            << lr << ","
                            << cf << ","
                            << result.rounds_to_target << ","
                            << result.final_accuracy << ","
                            << (result.reached_target ? "true" : "false") << "\n";
                    
                    std::cout << "Rounds to target: " << result.rounds_to_target 
                              << ", Final accuracy: " << result.final_accuracy << "\n";
                }
            }
        }
    }
    
    // Sort results by rounds_to_target (ascending) and accuracy (descending)
    std::sort(results.begin(), results.end(), 
              [](const OptimizationResult& a, const OptimizationResult& b) {
                  if (a.reached_target != b.reached_target) 
                      return a.reached_target > b.reached_target;
                  if (a.reached_target) 
                      return a.rounds_to_target < b.rounds_to_target;
                  return a.final_accuracy > b.final_accuracy;
              });
    
    return results;
}

float evaluate_test_set(FederatedClient &client, const std::vector<TrainingSample> &test_set)
{
    std::vector<std::vector<float>> predictions;
    std::vector<std::vector<float>> targets;

    // Get predictions for all test samples
    for (const auto &test_sample : test_set)
    {
        predictions.push_back(client.predict(test_sample.features));
        targets.push_back(test_sample.target);
    }

    // Calculate and return accuracy
    return Metrics::accuracy(predictions, targets);
}



OptimizationResult HyperParameterOptimizer::evaluate_parameters(
    const HyperParameters& params) {
    
    // Create federated learning setup with these parameters
    DataLoader loader("../data");
    auto dataset = loader.load_dataset("motion_metadata.csv");
    
    auto preprocessor = std::make_shared<DataPreprocessor>(seed);
    preprocessor->prepare_dataset(dataset);
    
    FederatedServer server(seed);
    std::vector<std::unique_ptr<FederatedClient>> clients;
    
    // Initialize 100 clients
    const size_t NUM_CLIENTS = 100;
    for (size_t i = 0; i < NUM_CLIENTS; i++) {
        clients.push_back(std::make_unique<FederatedClient>(
            params.topology.layers, preprocessor, seed + i));
    }
    
    float best_accuracy = 0.0f;
    size_t rounds_to_target = max_rounds;
    bool reached_target = false;
    
    // Run federated learning
    for (size_t round = 0; round < max_rounds; round++) {
        // Select clients for this round
        auto selected_clients = server.select_clients(clients.size(), params.client_fraction);
        
        // Train selected clients
        for (size_t client_idx : selected_clients) {
            for (size_t i = 0; i < params.samples_per_round; i++) {
                auto sample = preprocessor->get_next_training_sample(client_idx);
                clients[client_idx]->train_on_sample(
                    sample.features, sample.target, params.learning_rate);
            }
        }
        
        // Average weights
        std::vector<std::vector<float>> client_weights;
        for (size_t client_idx : selected_clients) {
            client_weights.push_back(clients[client_idx]->get_weights());
        }
        auto averaged_weights = server.average_weights(client_weights);
        
        // Update all clients
        for (auto& client : clients) {
            client->set_weights(averaged_weights);
        }
        
        // Evaluate every 10 rounds
        if (round % 10 == 0) {
            float accuracy = evaluate_test_set(*clients[0], preprocessor->get_test_set());
            best_accuracy = std::max(best_accuracy, accuracy);
            
            if (accuracy >= target_accuracy && !reached_target) {
                reached_target = true;
                rounds_to_target = round + 1;
                break;
            }
        }
    }
    
    return OptimizationResult{
        params,
        rounds_to_target,
        best_accuracy,
        reached_target
    };
}*/