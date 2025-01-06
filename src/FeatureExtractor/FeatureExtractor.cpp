#include "FeatureExtractor/FeatureExtractor.h"
#include <cmath>
#include <algorithm>
#include <numeric>

FeatureExtractor::FeatureExtractor() : buffer_size(256) {  // Using 256 samples like Arduino
    // Allocate FFTW buffers
    fft_in = fftwf_alloc_real(buffer_size);
    fft_out = fftwf_alloc_complex(buffer_size/2 + 1);
    
    // Create FFT plan
    fft_plan = fftwf_plan_dft_r2c_1d(buffer_size, fft_in, fft_out, FFTW_MEASURE);
}

FeatureExtractor::~FeatureExtractor() {
    fftwf_destroy_plan(fft_plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);
}

std::vector<float> FeatureExtractor::extract_features(const MotionSample& sample) {
    // We'll use x-axis acceleration for feature extraction like in Arduino code
    auto magnitudes = compute_fft_magnitudes(sample.acc_x);
    
    // Get frequency band features
    auto freq_features = calculate_frequency_bands(magnitudes);
    
    // Get statistical features
    auto stat_features = calculate_statistical_features(sample.acc_x);
    
    // Combine features
    std::vector<float> features;
    features.insert(features.end(), freq_features.begin(), freq_features.end());
    features.insert(features.end(), stat_features.begin(), stat_features.end());
    
    return features;
}

std::vector<float> FeatureExtractor::compute_fft_magnitudes(const std::vector<float>& input) {
    // Copy input data to FFT buffer
    for (size_t i = 0; i < buffer_size && i < input.size(); i++) {
        fft_in[i] = input[i];
    }
    
    // Apply Hamming window
    for (size_t i = 0; i < buffer_size; i++) {
        float window = 0.54f - 0.46f * std::cos(2 * M_PI * i / (buffer_size - 1));
        fft_in[i] *= window;
    }
    
    // Execute FFT
    fftwf_execute(fft_plan);
    
    // Compute magnitudes
    std::vector<float> magnitudes(buffer_size/2 + 1);
    for (size_t i = 0; i < buffer_size/2 + 1; i++) {
        float real = fft_out[i][0];
        float imag = fft_out[i][1];
        magnitudes[i] = std::sqrt(real*real + imag*imag);
    }
    
    return magnitudes;
}

std::vector<float> FeatureExtractor::calculate_frequency_bands(const std::vector<float>& magnitudes) {
    std::vector<float> band_energies(NUM_FREQ_BANDS);
    
    for (size_t band = 0; band < NUM_FREQ_BANDS; band++) {
        float band_energy = 0;
        size_t start_idx = (FREQ_BANDS[band] * magnitudes.size()) / SAMPLING_FREQ;
        size_t end_idx = (FREQ_BANDS[band + 1] * magnitudes.size()) / SAMPLING_FREQ;
        
        for (size_t i = start_idx; i < end_idx && i < magnitudes.size(); i++) {
            band_energy += magnitudes[i];
        }
        
        band_energies[band] = band_energy / (end_idx - start_idx);
    }
    
    return band_energies;
}

std::vector<float> FeatureExtractor::calculate_statistical_features(const std::vector<float>& signal) {
    std::vector<float> stats(3);
    
    // Mean
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    stats[0] = mean;
    
    // Max value
    stats[1] = *std::max_element(signal.begin(), signal.end());
    
    // Variance
    float variance = 0;
    for (const float& val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= signal.size();
    stats[2] = std::sqrt(variance);  // Standard deviation
    
    return stats;
}