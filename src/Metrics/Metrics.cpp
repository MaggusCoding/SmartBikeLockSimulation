#include "Metrics/Metrics.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <complex>

float Metrics::accuracy(const std::vector<std::vector<float>>& predictions, 
                       const std::vector<std::vector<float>>& targets) {
    auto pred_classes = get_predicted_classes(predictions);
    auto true_classes = get_true_classes(targets);
    
    int correct = 0;
    for (size_t i = 0; i < pred_classes.size(); i++) {
        if (pred_classes[i] == true_classes[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / pred_classes.size();
}

float Metrics::cross_entropy_loss(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets) {
    
    float total_loss = 0.0f;
    const float epsilon = 1e-15f; // To prevent log(0)
    
    for (size_t i = 0; i < predictions.size(); i++) {
        float sample_loss = 0.0f;
        for (size_t j = 0; j < predictions[i].size(); j++) {
            // Clip predictions to prevent numerical instability
            float pred = std::max(std::min(predictions[i][j], 1.0f - epsilon), epsilon);
            sample_loss -= targets[i][j] * std::log(pred);
        }
        total_loss += sample_loss;
    }
    
    return total_loss / predictions.size(); // Return average loss
}

std::array<std::array<int, 3>, 3> Metrics::confusion_matrix(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets) {
    
    std::array<std::array<int, 3>, 3> matrix = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    
    auto pred_classes = get_predicted_classes(predictions);
    auto true_classes = get_true_classes(targets);
    
    for (size_t i = 0; i < pred_classes.size(); i++) {
        matrix[true_classes[i]][pred_classes[i]]++;
    }
    
    return matrix;
}

std::array<float, 3> Metrics::roc_auc(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets) {
    
    std::array<float, 3> auc_scores = {0, 0, 0};
    
    // Calculate AUC for each class
    for (int class_idx = 0; class_idx < 3; class_idx++) {
        std::vector<std::pair<float, bool>> scores;
        
        // Collect predictions and true labels for this class
        for (size_t i = 0; i < predictions.size(); i++) {
            scores.push_back({
                predictions[i][class_idx],
                targets[i][class_idx] > 0.5f
            });
        }
        
        // Sort by predicted probability
        std::sort(scores.begin(), scores.end());
        
        // Calculate AUC
        float auc = 0;
        int pos_count = 0;
        int neg_count = 0;
        
        for (const auto& score : scores) {
            if (score.second) pos_count++;
            else neg_count++;
        }
        
        if (pos_count > 0 && neg_count > 0) {
            int tp = 0;
            int fp = 0;
            float prev_tpr = 0;
            float prev_fpr = 0;
            
            for (auto it = scores.rbegin(); it != scores.rend(); ++it) {
                if (it->second) tp++;
                else fp++;
                
                float tpr = static_cast<float>(tp) / pos_count;
                float fpr = static_cast<float>(fp) / neg_count;
                
                // Add trapezoid area
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2;
                
                prev_tpr = tpr;
                prev_fpr = fpr;
            }
        }
        
        auc_scores[class_idx] = auc;
    }
    
    return auc_scores;
}

std::array<float, 3> Metrics::f1_scores(const std::array<std::array<int, 3>, 3>& conf_matrix) {
    std::array<float, 3> f1_scores = {0, 0, 0};
    
    for (int i = 0; i < 3; i++) {
        int true_pos = conf_matrix[i][i];
        int false_pos = 0;
        int false_neg = 0;
        
        for (int j = 0; j < 3; j++) {
            if (i != j) {
                false_pos += conf_matrix[j][i];
                false_neg += conf_matrix[i][j];
            }
        }
        
        float precision = true_pos / static_cast<float>(true_pos + false_pos);
        float recall = true_pos / static_cast<float>(true_pos + false_neg);
        
        f1_scores[i] = 2 * (precision * recall) / (precision + recall);
    }
    
    return f1_scores;
}

void Metrics::print_confusion_matrix(const std::array<std::array<int, 3>, 3>& matrix) {
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\n";
    std::cout << "Actual ↓  ";
    
    // Column headers
    for (int i = 0; i < 3; i++) {
        std::cout << std::setw(8) << i;
    }
    std::cout << "\n";
    
    // Matrix values
    for (int i = 0; i < 3; i++) {
        std::cout << std::setw(8) << i;
        for (int j = 0; j < 3; j++) {
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << "\n";
    }
}

std::vector<int> Metrics::get_predicted_classes(const std::vector<std::vector<float>>& predictions) {
    std::vector<int> pred_classes;
    pred_classes.reserve(predictions.size());
    
    for (const auto& pred : predictions) {
        auto max_it = std::max_element(pred.begin(), pred.end());
        pred_classes.push_back(std::distance(pred.begin(), max_it));
    }
    
    return pred_classes;
}

std::vector<int> Metrics::get_true_classes(const std::vector<std::vector<float>>& targets) {
    std::vector<int> true_classes;
    true_classes.reserve(targets.size());
    
    for (const auto& target : targets) {
        auto max_it = std::max_element(target.begin(), target.end());
        true_classes.push_back(std::distance(target.begin(), max_it));
    }
    
    return true_classes;
}