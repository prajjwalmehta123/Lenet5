#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>
#include "activation.h"

// Constructor: Initialize any required data
Activation::Activation() {
    #ifdef USE_CUDA
    gpuImplementation = std::make_unique<ActivationGPU>();
    #endif
    // No specific initialization needed for ReLU
}

// Forward Propagation
std::vector<std::vector<float>> Activation::forwardProp(const std::vector<std::vector<float>>& input) {
    #ifdef USE_CUDA
    return gpuImplementation->forward(input);
    #endif
    size_t batch_size = input.size();
    size_t feature_size = input[0].size();
    inputImage = input; // Cache input for backpropagation

    std::vector<std::vector<float>> output(batch_size, std::vector<float>(feature_size));

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < feature_size; ++f) {
            output[b][f] = relu(input[b][f]);
        }
    }

    return output;
}

// Backward Propagation
std::vector<std::vector<float>> Activation::backProp(const std::vector<std::vector<float>>& dZ) {
    #ifdef USE_CUDA
    return gpuImplementation->backward(dZ);
    #endif
    size_t batch_size = dZ.size();
    size_t feature_size = dZ[0].size();
    std::vector<std::vector<float>> dA(batch_size, std::vector<float>(feature_size));

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < feature_size; ++f) {
            dA[b][f] = dZ[b][f] * d_relu(inputImage[b][f]);
        }
    }

    return dA;
}