#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "FCLayer.h"

// Default Constructor
FCLayer::FCLayer() {
    AdamOptimizer adam(0.001);
}

// Constructor
FCLayer::FCLayer(const std::pair<int, int>& weight_shape, const std::string& init_mode) {
    int rows = weight_shape.first;
    int cols = weight_shape.second;
    // Initialize weights and biases
    std::tie(weight, bias) = initialize(rows, cols);
    // std::cout<<"row"<<rows<<"cols"<<cols<<std::endl;
}

// Forward Propagation
std::vector<std::vector<float>> FCLayer::forward_prop(const std::vector<std::vector<float>>& input_array) {
    this->input_array = input_array; // Cache input for backpropagation

    int batch_size = input_array.size();
    int output_size = weight.size();
    int input_size = input_array[0].size();

    std::vector<std::vector<float>> output(batch_size, std::vector<float>(output_size, 0.0f));

    for (int i = 0; i < batch_size; ++i) {              // Loop over batch
        for (int j = 0; j < output_size; ++j) {         // Loop over outputs
            for (int k = 0; k < input_size; ++k) {      // Loop over inputs
                output[i][j] += input_array[i][k] * weight[j][k];
            }
            output[i][j] += bias[j]; // Add bias
        }
    }
    return output;
}

// Backward Propagation
std::vector<std::vector<float>> FCLayer::back_prop(const std::vector<std::vector<float>>& dZ) {
    int batch_size = dZ.size();
    int output_size = weight.size();       // Number of output neurons
    int input_size = weight[0].size();     // Number of input features

    // Initialize gradients
    dA_prev = std::vector<std::vector<float>>(batch_size, std::vector<float>(input_size, 0.0f));
    dW = std::vector<std::vector<float>>(output_size, std::vector<float>(input_size, 0.0f));
    db = std::vector<float>(output_size, 0.0f);

    // Compute gradients
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            db[j] += dZ[i][j];  // Gradient w.r.t biases

            for (int k = 0; k < input_size; ++k) {
                dW[j][k] += dZ[i][j] * input_array[i][k];        // Gradient w.r.t weights
                dA_prev[i][k] += dZ[i][j] * weight[j][k];        // Gradient w.r.t inputs
            }
        }
    }

    // Average gradients over the batch
    for (int j = 0; j < output_size; ++j) {
        db[j] /= static_cast<float>(batch_size);
        for (int k = 0; k < input_size; ++k) {
            dW[j][k] /= static_cast<float>(batch_size);
        }
    }

    // Update weights and biases using Adam optimizer
    adam.update_weight(weight, dW);
    adam.update_bias(bias, db);

    // Return gradient w.r.t inputs to propagate to previous layer
    return dA_prev;
}

// Initialize weights and biases
std::pair<std::vector<std::vector<float>>, std::vector<float>> FCLayer::initialize(int rows, int cols) {
    std::vector<std::vector<float>> w(rows, std::vector<float>(cols, 0.0f));
    std::vector<float> b(cols, 0.01f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 0.1f);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            w[i][j] = d(gen);
        }
    }
    return {w, b};
}