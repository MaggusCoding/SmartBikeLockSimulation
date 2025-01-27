#ifndef DATA_PREPROCESSOR_H
#define DATA_PREPROCESSOR_H

#include <vector>
#include <random>
#include "DataLoader/DataLoader.h"
#include "FeatureExtractor/FeatureExtractor.h"

struct TrainingSample {
    std::vector<float> features;
    std::vector<float> target;  // One-hot encoded target
};

class DataPreprocessor {
public:
    DataPreprocessor();
    explicit DataPreprocessor(uint32_t base_seed = 42);  // Base seed for reproducibility
    // Process all samples and prepare for training
    void prepare_dataset(const std::vector<MotionSample>& samples);
    TrainingSample get_next_training_sample(size_t client_id);  // Now requires client_id
    void reset_sampling();  
    // Get training batches
    std::vector<TrainingSample> get_training_batch(size_t batch_size);
    std::vector<TrainingSample> get_balanced_batch(size_t samples_per_class);
    
    // Get test set
    std::vector<TrainingSample> get_test_set() const { return test_set; }
    
    // Scaling parameters for future use
    std::vector<float> get_scale_params() const { 
        return {feature_min, feature_max}; 
    }

private:
    std::vector<TrainingSample> training_set;
    std::vector<TrainingSample> test_set;
    
    float feature_min;
    float feature_max;
    
    FeatureExtractor feature_extractor;
    std::mt19937 rng;
    
    // Helper methods
    std::vector<float> create_one_hot_encoding(int label);
    void normalize_features(std::vector<float>& features);
    void split_train_test(std::vector<TrainingSample>& all_samples, float test_ratio = 0.2);
    uint32_t base_seed;  // Store base seed for reset functionality
    std::unordered_map<size_t, std::mt19937> client_rngs;  // RNG per client
    std::unordered_map<size_t, std::vector<size_t>> client_shuffled_indices;  // Indices per client
    std::unordered_map<size_t, size_t> client_current_indices;  // Current position per client
};


#endif