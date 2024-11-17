#include "lenet5.h"
#include <iostream>
#include <random>

// Constructor
LeNet5::LeNet5(): F6({120, 84}, "Gaussian_dist") {

    // Initialize F6 after kernel_shape is fully initialized
    // F6 = FCLayer({kernel_shape["F6"][0], kernel_shape["F6"][1]});

    for (const auto& pair : kernel_shape) {
        std::cout << pair.first << ": ";
        for (const auto& val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Print hyperparameters for convolutional layer
    std::cout << "\nHyperparameters (Convolutional Layer):" << std::endl;
    for (const auto& pair : hparameters_convlayer) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Print hyperparameters for pooling
    std::cout << "\nHyperparameters (Pooling):" << std::endl;
    for (const auto& pair : hparameters_pooling) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
}

// Forward Propagation
auto LeNet5::Forward_Propagation(const std::vector<float>& input_image, const std::vector<float>& input_label, const std::string& mode) {
    // label = input_label; // Cache label for backpropagation
    // int out = 1;         // Placeholder return value
    // return out;
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