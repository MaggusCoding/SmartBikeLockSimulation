#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <array>
#include <string>

class Metrics {
public:
    // Calculate accuracy
    static float accuracy(const std::vector<std::vector<float>>& predictions, 
                         const std::vector<std::vector<float>>& targets);
    
    // Calculate confusion matrix
    static std::array<std::array<int, 3>, 3> confusion_matrix(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets);
    
    // Calculate ROC AUC score for each class
    static std::array<float, 3> roc_auc(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets);
    
    // Calculate F1 score for each class
    static std::array<float, 3> f1_scores(
        const std::array<std::array<int, 3>, 3>& conf_matrix);
    
    // Pretty print confusion matrix
    static void print_confusion_matrix(const std::array<std::array<int, 3>, 3>& matrix);

    // Calculate cross-entropy loss
    static float cross_entropy_loss(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets);
        
    
private:
    // Convert probabilities to class predictions
    static std::vector<int> get_predicted_classes(const std::vector<std::vector<float>>& predictions);
    static std::vector<int> get_true_classes(const std::vector<std::vector<float>>& targets);
};

#endif