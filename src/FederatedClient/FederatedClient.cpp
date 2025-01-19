#include "FederatedClient.h"
#include "Metrics/Metrics.h"
#include <iostream>
#include <random>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>

class FederatedClient {
public:
    FederatedClient(const std::vector<size_t>& topology, 
                   const std::string& data_path,
                   const std::string& client_id)
        : client_id(client_id), 
          rng(std::random_device{}()),
          data_loader(data_path),
          motion_data_path(data_path + "/motion_data") {
    
        // Initialize neural network with given topology
        network = std::make_unique<NeuralNetwork>(topology);
        
        // Load metadata to get available samples
        dataset = data_loader.load_dataset("motion_metadata.csv");
        
        std::cout << "Initialized client " << client_id 
                  << " with access to " << dataset.size() << " samples\n";
    }
    
    // Online learning: train on a single randomly selected sample from files
    void train_single_sample(float learning_rate) {
        if (dataset.empty()) return;
        
        // Randomly select a sample from metadata
        std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
        const auto& sample = dataset[dist(rng)];
        
        // Extract features
        auto features = feature_extractor.extract_features(sample);
        
        // Create target vector
        std::vector<float> target(3, 0.0f);  // 3 classes
        target[sample.label] = 1.0f;
        
        // Train on single sample
        auto output = network->forward(features);
        network->train(features, target, learning_rate);
        
        // Calculate error for monitoring
        float error = 0.0f;
        for (size_t i = 0; i < output.size(); i++) {
            float diff = output[i] - target[i];
            error += diff * diff;
        }
        error /= output.size();
        
        if (++training_steps % 100 == 0) {  // Log less frequently
            std::cout << "Client " << client_id << " training step " << training_steps 
                      << " error: " << std::scientific << error 
                      << " (Sample: " << sample.filename << ")\n";
        }
    }
    
    std::vector<float> get_weights() const {
        return network->get_flat_weights();
    }
    
    void set_weights(const std::vector<float>& weights) {
        network->set_flat_weights(weights);
    }
    
    float evaluate() const {
        // Create a temporary feature extractor for evaluation
        FeatureExtractor eval_feature_extractor;
        
        // Evaluate on a random subset of samples to simulate Arduino's memory constraints
        const size_t EVAL_SAMPLES = 100;
        std::vector<std::vector<float>> predictions;
        std::vector<std::vector<float>> targets;
        
        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (size_t i = 0; i < std::min(EVAL_SAMPLES, dataset.size()); i++) {
            const auto& sample = dataset[indices[i]];
            auto features = eval_feature_extractor.extract_features(sample);
            
            std::vector<float> target(3, 0.0f);
            target[sample.label] = 1.0f;
            
            predictions.push_back(network->forward(features));
            targets.push_back(target);
        }
        
        return Metrics::accuracy(predictions, targets);
    }

private:
    std::string client_id;
    std::string motion_data_path;
    std::unique_ptr<NeuralNetwork> network;
    DataLoader data_loader;
    FeatureExtractor feature_extractor;
    std::vector<MotionSample> dataset;  // Metadata only
    std::mt19937 rng;
    size_t training_steps = 0;
    
    // Helper method to get available motion files
    std::vector<std::string> get_motion_files() const {
        std::vector<std::string> files;
        for (const auto& entry : std::filesystem::directory_iterator(motion_data_path)) {
            if (entry.path().extension() == ".csv") {
                files.push_back(entry.path().filename().string());
            }
        }
        return files;
    }
};