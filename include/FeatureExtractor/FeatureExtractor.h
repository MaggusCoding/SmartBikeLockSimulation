#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <complex>
#include <fftw3.h>
#include "DataLoader/DataLoader.h"

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();
    
    // Extract features from a motion sample
    std::vector<float> extract_features(const MotionSample& sample);
    
private:
    // FFT processing
    std::vector<float> compute_fft_magnitudes(const std::vector<float>& input);
    
    // Feature calculation
    std::vector<float> calculate_frequency_bands(const std::vector<float>& magnitudes);
    std::vector<float> calculate_statistical_features(const std::vector<float>& signal);
    
    // FFTW plans and buffers
    fftwf_plan fft_plan;
    float* fft_in;
    fftwf_complex* fft_out;
    size_t buffer_size;
    
    // Constants matching your Arduino implementation
    static constexpr size_t NUM_FREQ_BANDS = 8;
    static constexpr float FREQ_BANDS[9] = {0, 5, 10, 15, 20, 25, 30, 40, 50};
    static constexpr float SAMPLING_FREQ = 100;  // Hz
    static constexpr size_t FEATURE_BINS = 8;
    static constexpr size_t TOTAL_FEATURES = 11;  // 8 frequency bins + 3 statistical features
};

#endif