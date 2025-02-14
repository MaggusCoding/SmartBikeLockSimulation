/**#pragma once

#include <vector>
#include <array>
#include <string>
#include <cstdint>

struct NetworkTopology {
    std::vector<size_t> layers;
    std::string to_string() const {
        std::string result = "topology_";
        for (size_t layer : layers) {
            result += std::to_string(layer) + "_";
        }
        return result;
    }
};

struct HyperParameters {
    size_t samples_per_round;
    float learning_rate;
    NetworkTopology topology;
    float client_fraction;
    
    std::string to_string() const {
        return "spr_" + std::to_string(samples_per_round) + 
               "_lr_" + std::to_string(learning_rate) +
               "_cf_" + std::to_string(client_fraction) + 
               "_" + topology.to_string();
    }
};

struct OptimizationResult {
    HyperParameters params;
    size_t rounds_to_target;
    float final_accuracy;
    bool reached_target;
};

class HyperParameterOptimizer {
public:
    HyperParameterOptimizer(float target_accuracy = 0.90f, 
                           size_t max_rounds = 2000,
                           uint32_t seed = 42)
        : target_accuracy(target_accuracy)
        , max_rounds(max_rounds)
        , seed(seed) {
        initialize_search_space();
    }
    
    std::vector<OptimizationResult> run_optimization();
    
private:
    void initialize_search_space() {
        // Network topologies to try (input and output layers fixed)
        topologies = {
            NetworkTopology{{11, 35, 3}},      // Smaller network
            NetworkTopology{{11, 50, 3}},     // Medium network
            NetworkTopology{{11, 65, 3}},    // Larger network
        };
        
        // Training samples per round
        samples_per_round = {20, 25, 30};
        
        // Learning rates
        learning_rates = {0.4f, 0.5f, 0.6f};
        
        // Client fractions
        client_fractions = {0.1f, 0.2f};
    }
    
    OptimizationResult evaluate_parameters(const HyperParameters& params);
    
    float target_accuracy;
    size_t max_rounds;
    uint32_t seed;
    
    // Search space
    std::vector<NetworkTopology> topologies;
    std::vector<size_t> samples_per_round;
    std::vector<float> learning_rates;
    std::vector<float> client_fractions;
};*/