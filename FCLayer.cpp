#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include "FCLayer.h"

// Default Constructor
FCLayer::FCLayer() {
    AdamOptimizer adam(0.01, 0.9, 0.99, 1e-8);
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
    int input_size = input_array[0].size();

    std::vector<std::vector<float>> output(batch_size, std::vector<float>(output_size, 0.0f));

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = bias[j]; // Start with bias
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < input_size; ++k) {
                sum += input_array[i][k] * weight[j][k];
            }
            output[i][j] = sum;
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
    #pragma omp parallel
    {
        // Thread-local storage for partial gradient accumulations
        std::vector<std::vector<float>> local_dW(output_size, std::vector<float>(input_size, 0.0f));
        std::vector<float> local_db(output_size, 0.0f);

        #pragma omp for
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                local_db[j] += dZ[i][j];
                for (int k = 0; k < input_size; ++k) {
                    local_dW[j][k] += dZ[i][j] * input_array[i][k];
                    #pragma omp atomic
                    dA_prev[i][k] += dZ[i][j] * weight[j][k];
                }
            }
        }

        // Merge local gradients into global ones
        #pragma omp critical
        {
            for (int j = 0; j < output_size; ++j) {
                db[j] += local_db[j];
                for (int k = 0; k < input_size; ++k) {
                    dW[j][k] += local_dW[j][k];
                }
            }
        }
    }

    // Average gradients over the batch
    #pragma omp parallel for
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

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            w[i][j] = d(gen);
        }
    }
    return {w, b};
}