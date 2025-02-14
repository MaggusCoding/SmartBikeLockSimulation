#include "DataPreprocessor/DataPreprocessor.h"
#include <algorithm>
#include <numeric>
#include <random>

// Constructor now takes a seed parameter for reproducibility
DataPreprocessor::DataPreprocessor(uint32_t seed) : 
    feature_min(0), 
    feature_max(1),
    rng(seed),
    base_seed(base_seed) {
}

void DataPreprocessor::prepare_dataset(const std::vector<MotionSample>& samples) {
    // Previous feature extraction and normalization code remains the same
    std::vector<TrainingSample> all_samples;
    
    for (const auto& sample : samples) {
        TrainingSample training_sample;
        training_sample.features = feature_extractor.extract_features(sample);
        training_sample.target = create_one_hot_encoding(sample.label);
        all_samples.push_back(training_sample);
    }
    
    feature_min = std::numeric_limits<float>::max();
    feature_max = std::numeric_limits<float>::lowest();
    
    for (const auto& sample : all_samples) {
        for (float feature : sample.features) {
            feature_min = std::min(feature_min, feature);
            feature_max = std::max(feature_max, feature);
        }
    }
    
    for (auto& sample : all_samples) {
        normalize_features(sample.features);
    }
    
    split_train_test(all_samples);
}

std::vector<float> DataPreprocessor::create_one_hot_encoding(int label) {
    std::vector<float> encoding(3, 0.0f);  // 3 classes
    encoding[label] = 1.0f;
    return encoding;
}

void DataPreprocessor::normalize_features(std::vector<float>& features) {
    float range = feature_max - feature_min;
    if (range > 0) {
        for (float& feature : features) {
            feature = (feature - feature_min) / range;
        }
    }
}

void DataPreprocessor::split_train_test(std::vector<TrainingSample>& all_samples, float test_ratio) {
    std::shuffle(all_samples.begin(), all_samples.end(), rng);
    
    size_t test_size = static_cast<size_t>(all_samples.size() * test_ratio);
    test_set.assign(all_samples.begin(), all_samples.begin() + test_size);
    training_set.assign(all_samples.begin() + test_size, all_samples.end());
}

TrainingSample DataPreprocessor::get_next_training_sample(size_t client_id) {
    if (training_set.empty()) {
        throw std::runtime_error("No training samples available");
    }
    
    // Initialize client-specific RNG and indices if not exists
    if (client_rngs.find(client_id) == client_rngs.end()) {
        // Create deterministic seed for this client using base_seed
        uint32_t client_seed = base_seed + client_id;
        client_rngs[client_id] = std::mt19937(client_seed);
        
        // Initialize shuffled indices for this client
        client_shuffled_indices[client_id].resize(training_set.size());
        std::iota(client_shuffled_indices[client_id].begin(), 
                 client_shuffled_indices[client_id].end(), 0);
        std::shuffle(client_shuffled_indices[client_id].begin(), 
                    client_shuffled_indices[client_id].end(), 
                    client_rngs[client_id]);
        
        client_current_indices[client_id] = 0;
    }
    
    // Get sample using client's current position
    size_t index = client_shuffled_indices[client_id][client_current_indices[client_id]];
    TrainingSample sample = training_set[index];
    
    // Update client's position
    client_current_indices[client_id] = 
        (client_current_indices[client_id] + 1) % training_set.size();
    
    // Reshuffle this client's indices if we've gone through all samples
    if (client_current_indices[client_id] == 0) {
        std::shuffle(client_shuffled_indices[client_id].begin(), 
                    client_shuffled_indices[client_id].end(), 
                    client_rngs[client_id]);
    }
    
    return sample;
}

void DataPreprocessor::reset_sampling() {
    // Clear all client states
    client_rngs.clear();
    client_shuffled_indices.clear();
    client_current_indices.clear();
}