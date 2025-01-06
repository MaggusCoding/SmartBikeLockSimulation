#include "NeuralNetwork/NeuralNetwork.h"
#include "DataLoader/DataLoader.h"
#include "FeatureExtractor/FeatureExtractor.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include "Metrics/Metrics.h"
#include <iostream>
#include <iomanip>

void print_training_sample(const TrainingSample& sample) {
    std::cout << "Features: ";
    for (float f : sample.features) {
        std::cout << std::fixed << std::setprecision(4) << f << " ";
    }
    std::cout << "\nTarget: ";
    for (float t : sample.target) {
        std::cout << std::fixed << std::setprecision(0) << t << " ";
    }
    std::cout << "\n";
}

void evaluate_model(NeuralNetwork& nn, const std::vector<TrainingSample>& test_set) {
    std::vector<std::vector<float>> predictions;
    std::vector<std::vector<float>> targets;
    
    // Get predictions for test set
    for (const auto& sample : test_set) {
        predictions.push_back(nn.forward(sample.features));
        targets.push_back(sample.target);
    }
    
    // Calculate metrics
    float acc = Metrics::accuracy(predictions, targets);
    auto conf_matrix = Metrics::confusion_matrix(predictions, targets);
    auto auc_scores = Metrics::roc_auc(predictions, targets);
    auto f1 = Metrics::f1_scores(conf_matrix);
    
    std::cout << "\nModel Evaluation:\n";
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << acc << "\n";
    
    Metrics::print_confusion_matrix(conf_matrix);
    
    std::cout << "\nROC AUC Scores:\n";
    std::cout << "Class 0: " << auc_scores[0] << "\n";
    std::cout << "Class 1: " << auc_scores[1] << "\n";
    std::cout << "Class 2: " << auc_scores[2] << "\n";
    
    std::cout << "\nF1 Scores:\n";
    std::cout << "Class 0: " << f1[0] << "\n";
    std::cout << "Class 1: " << f1[1] << "\n";
    std::cout << "Class 2: " << f1[2] << "\n";
}

int main() {
    try {
        // Load dataset
        DataLoader loader("../data");
        auto dataset = loader.load_dataset("motion_metadata.csv");
        std::cout << "Loaded " << dataset.size() << " samples\n\n";
        
        // Prepare data for training
        DataPreprocessor preprocessor;
        preprocessor.prepare_dataset(dataset);
        
        // Get test set early
        auto test_set = preprocessor.get_test_set();
        std::cout << "Test set size: " << test_set.size() << " samples\n";
        
        // Create neural network
        std::vector<size_t> topology = {11, 50, 60, 3};  // Matching your Arduino implementation
        NeuralNetwork nn(topology);
        
        std::cout << "\nTraining configuration:\n";
        std::cout << "Input features: " << topology[0] << "\n";
        std::cout << "Hidden layer 1: " << topology[1] << " neurons\n";
        std::cout << "Hidden layer 2: " << topology[2] << " neurons\n";
        std::cout << "Output layer: " << topology[3] << " neurons\n\n";
        
        // Training parameters
        const int EPOCHS = 2000;  // Increased epochs
        const size_t SAMPLES_PER_CLASS = 4;  // We'll get 4 samples per class = 12 total
        const int EVAL_INTERVAL = 100;
        float learning_rate = 0.01f;  // Reduced initial learning rate
        
        std::cout << "Starting training...\n";
        float best_accuracy = 0.0f;
        int epochs_without_improvement = 0;
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Get balanced batch
            auto batch = preprocessor.get_balanced_batch(SAMPLES_PER_CLASS);
            float epoch_error = 0.0f;
            
            // Train on batch
            for (const auto& sample : batch) {
                auto output = nn.forward(sample.features);
                nn.train(sample.features, sample.target, learning_rate);
                
                // Calculate error
                float error = 0.0f;
                for (size_t i = 0; i < output.size(); i++) {
                    float diff = output[i] - sample.target[i];
                    error += diff * diff;
                }
                epoch_error += error / output.size();
            }
            
            epoch_error /= batch.size();
            
            // Periodic evaluation
            if (epoch % EVAL_INTERVAL == 0) {
                std::cout << "\nEpoch " << epoch << ":\n";
                std::cout << "Training error: " << std::scientific << epoch_error << "\n";
                
                // Evaluate on test set
                std::vector<std::vector<float>> predictions;
                std::vector<std::vector<float>> targets;
                for (const auto& sample : test_set) {
                    predictions.push_back(nn.forward(sample.features));
                    targets.push_back(sample.target);
                }
                float current_accuracy = Metrics::accuracy(predictions, targets);
                
                // Track best model and implement early stopping
                if (current_accuracy > best_accuracy) {
                    best_accuracy = current_accuracy;
                    epochs_without_improvement = 0;
                    std::cout << "New best accuracy: " << current_accuracy << "\n";
                } else {
                    epochs_without_improvement++;
                    if (epochs_without_improvement >= 5) {  // No improvement for 500 epochs
                        learning_rate *= 0.5f;
                        std::cout << "\nReducing learning rate to: " << learning_rate << "\n";
                        epochs_without_improvement = 0;
                    }
                }
                
                evaluate_model(nn, test_set);
            }
        }
        
        // Final evaluation
        std::cout << "\nTraining completed. Final evaluation:\n";
        evaluate_model(nn, test_set);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}