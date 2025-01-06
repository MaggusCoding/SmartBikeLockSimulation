#include "DataPreprocessor/DataPreprocessor.h"
#include <algorithm>
#include <numeric>
#include <random>

DataPreprocessor::DataPreprocessor() : 
    feature_min(0), 
    feature_max(1),
    rng(std::random_device{}()) {}



std::vector<TrainingSample> DataPreprocessor::get_balanced_batch(size_t samples_per_class) {
    // Split training samples by class
    std::vector<std::vector<TrainingSample>> samples_by_class(3);
    for (const auto& sample : training_set) {
        for (size_t i = 0; i < sample.target.size(); i++) {
            if (sample.target[i] > 0.5f) {
                samples_by_class[i].push_back(sample);
                break;
            }
        }
    }
    
    // Create balanced batch
    std::vector<TrainingSample> batch;
    for (size_t class_idx = 0; class_idx < 3; class_idx++) {
        auto& class_samples = samples_by_class[class_idx];
        std::vector<size_t> indices(class_samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (size_t i = 0; i < samples_per_class && i < indices.size(); i++) {
            batch.push_back(class_samples[indices[i]]);
        }
    }
    
    // Final shuffle of the balanced batch
    std::shuffle(batch.begin(), batch.end(), rng);
    return batch;
}

void DataPreprocessor::prepare_dataset(const std::vector<MotionSample>& samples) {
    std::vector<TrainingSample> all_samples;
    
    // Extract features from all samples
    for (const auto& sample : samples) {
        TrainingSample training_sample;
        training_sample.features = feature_extractor.extract_features(sample);
        training_sample.target = create_one_hot_encoding(sample.label);
        all_samples.push_back(training_sample);
    }
    
    // Find global min/max for feature normalization
    feature_min = std::numeric_limits<float>::max();
    feature_max = std::numeric_limits<float>::lowest();
    
    for (const auto& sample : all_samples) {
        for (float feature : sample.features) {
            feature_min = std::min(feature_min, feature);
            feature_max = std::max(feature_max, feature);
        }
    }
    
    // Normalize all features
    for (auto& sample : all_samples) {
        normalize_features(sample.features);
    }
    
    // Split into training and test sets
    split_train_test(all_samples);
}

std::vector<TrainingSample> DataPreprocessor::get_training_batch(size_t batch_size) {
    std::vector<TrainingSample> batch;
    batch_size = std::min(batch_size, training_set.size());
    
    std::vector<size_t> indices(training_set.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (size_t i = 0; i < batch_size; i++) {
        batch.push_back(training_set[indices[i]]);
    }
    
    return batch;
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