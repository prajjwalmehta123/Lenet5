#include "lenet5.h"
#include <iostream>
#include <random>
#include "conv.h"

// Constructor
LeNet5::LeNet5(){
    for (const auto& pair : kernel_shape) {
        std::cout << pair.first << ": ";
        for (const auto& val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    int stride = 1;
    int padding = 0;
    ConvolutionLayer convLayer(kernel_shape["C1"][2], kernel_shape["C1"][3], kernel_shape["C1"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    int imageHeight = 32;  // 32x32 images (already padded)
    int imageWidth = 32;
    F6 = FCLayer({kernel_shape["F6"][0], kernel_shape["F6"][1]});
    
}

std::vector<std::vector<float>> LeNet5::flattenTensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& a3_FP) {
    size_t batchSize = a3_FP.size();
    size_t channels = a3_FP[0][0][0].size();
    std::vector<std::vector<float>> flattened(batchSize, std::vector<float>(channels));

    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            flattened[i][c] = a3_FP[i][0][0][c];
        }
    }
    return flattened;
}

// Forward Propagation
void LeNet5::Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels) {
    std::vector<std::vector<float>> outputBatch = convLayer.forward(batch_images, imageHeight, imageWidth);
    // std::vector<std::vector<float>> flattened = flattenTensor(a3_FP);
    // F6.forward_prop(flattened);
}

// Back Propagation
void LeNet5::Back_Propagation(float momentum, float weight_decay) {
    // Placeholder for backpropagation logic
}

// Stochastic Diagonal Levenberg-Marquardt (SDLM)
void LeNet5::SDLM(float mu, float lr_global) {
    // Placeholder for SDLM logic
}

// Initialize weights
std::pair<std::vector<std::vector<float>>, std::vector<float>> LeNet5::initialize_weights(std::vector<int> kernel_shape) {
    std::vector<std::vector<float>> weight;
    std::vector<float> bias;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0, 0.1); // Mean = 0.0, Std Dev = 0.1

    // Initialize weight matrix (rows = kernel_shape[0], cols = kernel_shape[1])
    int rows = kernel_shape[0];
    int cols = kernel_shape[1];
    weight.resize(rows, std::vector<float>(cols, 0.0f)); // Initialize with zeros
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weight[i][j] = d(gen); // Assign random values from Gaussian distribution
        }
    }

    // Initialize bias vector (size = kernel_shape[1])
    bias.resize(cols, 0.01f); // Small constant value for bias initialization

    return {weight, bias};
}