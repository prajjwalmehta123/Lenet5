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
}

// Forward Propagation
std::vector<std::vector<float>> FCLayer::forward_prop(const std::vector<std::vector<float>>& input_array) {
    this->input_array = input_array; // Cache input for backpropagation

    int batch_size = input_array.size();
    int output_size = weight.size();

    // Compute output = input_array * weight + bias
    std::vector<std::vector<float>> output(batch_size, std::vector<float>(output_size, 0.0f));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            for (int k = 0; k < weight.size(); ++k) {
                output[i][j] += input_array[i][k] * weight[k][j];
            }
            output[i][j] += bias[j]; // Add bias
        }
    }
    return output;
}

// Backward Propagation
std::vector<std::vector<float>> FCLayer::back_prop(const std::vector<std::vector<float>>& dZ, float momentum, float weight_decay) {
    int batch_size = dZ.size();
    int input_size = weight.size();
    int output_size = weight[0].size();

    // Compute dA = dZ * weight^T
    std::vector<std::vector<float>> dA(batch_size, std::vector<float>(input_size, 0.0f));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            for (int k = 0; k < output_size; ++k) {
                dA[i][j] += dZ[i][k] * weight[j][k];
            }
        }
    }

    // Compute dW = input_array^T * dZ
    std::vector<std::vector<float>> dW(input_size, std::vector<float>(output_size, 0.0f));
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            for (int k = 0; k < batch_size; ++k) {
                dW[i][j] += input_array[k][i] * dZ[k][j];
            }
        }
    }

    // Compute db = sum of dZ across batches
    std::vector<float> db(output_size, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            db[j] += dZ[i][j];
        }
    }
    // Update weights and biases using Adam
    adam.update(weight, dW);
    adam.update(bias, db);

    return dA;
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