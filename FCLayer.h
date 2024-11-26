// FCLayer.h
#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>
#include <string>
#include <utility>
#include <random>
#include <tuple>
#include "adam.h"


#ifdef USE_CUDA
#include "fc_gpu.cuh"
#include <memory>
#endif

class FCLayer {
private:
#ifdef USE_CUDA
    std::unique_ptr<FCLayerGPU> gpuImplementation;
#endif
    std::vector<std::vector<float>> weight;
    std::vector<float> bias;
    AdamOptimizer adam;
    std::vector<std::vector<float>> input_array;
    std::vector<std::vector<float>> dW;
    std::vector<float> db;
    std::vector<std::vector<float>> dA_prev;
    // Helper function to initialize weights and biases
    std::pair<std::vector<std::vector<float>>, std::vector<float>> initialize(
        int rows, int cols);

public:
    FCLayer();
    FCLayer(const std::pair<int, int>& weight_shape, const std::string& init_mode = "Gaussian_dist");
    std::vector<std::vector<float>> getWeights() const;
    std::vector<float> getBiases() const;
    
    // Move constructor and assignment operator for proper unique_ptr handling
    FCLayer(FCLayer&& other) noexcept;
    FCLayer& operator=(FCLayer&& other) noexcept;
    
    // Delete copy constructor and assignment
    FCLayer(const FCLayer&) = delete;
    FCLayer& operator=(const FCLayer&) = delete;

    std::vector<std::vector<float>> forward_prop(const std::vector<std::vector<float>>& input_array);
    std::vector<std::vector<float>> back_prop(const std::vector<std::vector<float>>& dZ);
};

#endif // FCLAYER_H