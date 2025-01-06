# src/main.cpp

#include "NeuralNetworkBikeLock.h"
#include "DataLoader.h"
#include <iostream>
#include <memory>

class BikeNode {
public:
    BikeNode() {
        nn = std::make_unique<NeuralNetworkBikeLock>();
        nn->init(NNConfig::LAYERS, nullptr, NNConfig::NUM_LAYERS);
    }

    void trainOnMotionData(const std::vector<float>& features, int label) {
        nn->performLiveTraining(features.data(), label);
    }

    std::vector<float> getWeights() {
        std::vector<float> weights(NNConfig::MAX_WEIGHTS);
        nn->getWeights(weights.data(), weights.size());
        return weights;
    }

    void updateWeights(const std::vector<float>& weights) {
        nn->updateNetworkWeights(weights.data(), weights.size());
    }

private:
    std::unique_ptr<NeuralNetworkBikeLock> nn;
};

int main() {
    DataLoader loader("./data");
    BikeNode node;
    
    // Load metadata
    auto metadata = loader.loadMetadata("motion_metadata.csv");
    std::cout << "Loaded " << metadata.size() << " motion samples\n";
    
    // Process each motion file
    for (const auto& entry : metadata) {
        auto motionData = loader.loadMotionData(entry.filename);
        auto features = loader.extractFeatures(motionData);
        
        node.trainOnMotionData(features, entry.label);
    }
    
    auto weights = node.getWeights();
    std::cout << "Training complete. Model has " << weights.size() << " weights\n";
    
    return 0;
}