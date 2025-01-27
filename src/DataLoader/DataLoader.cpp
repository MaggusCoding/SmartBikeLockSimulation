#include "DataLoader/DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

DataLoader::DataLoader(const std::string& base_path) 
    : base_path(base_path), 
      motion_data_path(base_path + "/motion_data") {}

std::vector<MotionSample> DataLoader::load_dataset(const std::string& metadata_file) {
    std::vector<MotionSample> dataset;
    std::ifstream file(base_path + "/" + metadata_file);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open metadata file: " + metadata_file);
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    // Read each line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        
        // Parse CSV fields
        std::getline(ss, field, ',');
        int sample_id = std::stoi(field);
        
        std::string timestamp;
        std::getline(ss, timestamp, ',');
        
        std::getline(ss, field, ',');
        int label = std::stoi(field);
        
        std::string filename;
        std::getline(ss, filename, ',');
        
        try {
            // Load the corresponding motion data file
            dataset.push_back(load_motion_file(filename, sample_id, timestamp, label));
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << filename << ": " << e.what() << "\n";
        }
    }
    
    return dataset;
}

MotionSample DataLoader::load_motion_file(const std::string& filename, 
                                        int sample_id, 
                                        const std::string& timestamp, 
                                        int label) {
    MotionSample sample;
    sample.sample_id = sample_id;
    sample.timestamp = timestamp;
    sample.label = label;
    sample.filename = filename;
    
    std::ifstream file(motion_data_path + "/" + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open motion file: " + filename);
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        
        // Skip timestamp
        std::getline(ss, field, ',');
        
        // Read accelerometer data
        std::getline(ss, field, ',');
        sample.acc_x.push_back(std::stof(field));
        
        std::getline(ss, field, ',');
        sample.acc_y.push_back(std::stof(field));
        
        std::getline(ss, field, ',');
        sample.acc_z.push_back(std::stof(field));
    }
    
    return sample;
}

std::unordered_map<int, int> DataLoader::get_label_distribution() const {
    return std::unordered_map<int, int>();  // TODO: Implement this
}