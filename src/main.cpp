#include "NeuralNetwork/NeuralNetwork.h"
#include "DataLoader/DataLoader.h"
#include "FeatureExtractor/FeatureExtractor.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include "Metrics/Metrics.h"
#include "FederatedClient/FederatedClient.h"
#include "FederatedServer/FederatedServer.h"
#include "HPO/HyperParameterOptimizer.h"
#include <iostream>
#include <iomanip>

int main() {
    try {
        std::cout << "Starting Hyperparameter Optimization...\n";
        
        HyperParameterOptimizer optimizer(
            0.90f,  // target accuracy
            2000,   // max rounds
            42      // seed
        );
        
        auto results = optimizer.run_optimization();
        
        // Print best results
        std::cout << "\nTop 3 parameter combinations:\n";
        for (size_t i = 0; i < std::min(size_t(3), results.size()); i++) {
            const auto& result = results[i];
            std::cout << "\n" << (i+1) << ". Configuration:\n"
                      << "Network: " << result.params.topology.to_string() << "\n"
                      << "Samples per round: " << result.params.samples_per_round << "\n"
                      << "Learning rate: " << result.params.learning_rate << "\n"
                      << "Client fraction: " << result.params.client_fraction << "\n"
                      << "Rounds to target: " << result.rounds_to_target << "\n"
                      << "Final accuracy: " << (result.final_accuracy * 100.0f) << "%\n"
                      << "Reached target: " << (result.reached_target ? "Yes" : "No") << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}