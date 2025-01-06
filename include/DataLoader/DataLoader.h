#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>

struct MotionSample {
    int sample_id;
    std::string timestamp;
    int label;
    std::string filename;
    std::vector<float> acc_x;
    std::vector<float> acc_y;
    std::vector<float> acc_z;
};

class DataLoader {
public:
    DataLoader(const std::string& base_path);
    
    // Load all data
    std::vector<MotionSample> load_dataset(const std::string& metadata_file);
    
    // Load individual files
    MotionSample load_motion_file(const std::string& filename, int sample_id, 
                                 const std::string& timestamp, int label);
    
    // Get statistics about the dataset
    std::unordered_map<int, int> get_label_distribution() const;
    
private:
    std::string base_path;
    std::string motion_data_path;
};

#endif