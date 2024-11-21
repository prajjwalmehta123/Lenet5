// test_adam_optimizer.cpp

#include "adam.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    // Define Adam optimizer parameters
    float learning_rate = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Instantiate the Adam optimizer
    AdamOptimizer optimizer(learning_rate, beta1, beta2, epsilon);

    // Define sample weights and biases
    std::vector<std::vector<float>> weights = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    std::vector<float> biases = {0.5f, -0.5f};

    // Define sample gradients for weights and biases
    std::vector<std::vector<float>> weight_gradients = {
        {0.1f, 0.2f},
        {0.3f, 0.4f}
    };

    std::vector<float> bias_gradients = {0.05f, -0.05f};

    // Perform first update (t=1)
    optimizer.update_weight(weights, weight_gradients);
    optimizer.update_bias(biases, bias_gradients);

    // Expected updates based on Adam algorithm for t=1
    // Since t=1, bias-corrected moments are m_hat = m / (1 - beta1^1) = m / 0.1
    // Similarly, v_hat = v / (1 - beta2^1) = v / 0.001
    // Given initial m and v are zero, after first update:
    // m = beta1 * 0 + (1 - beta1) * g = 0.1 * g
    // v = beta2 * 0 + (1 - beta2) * g^2 = 0.001 * g^2
    // m_hat = (0.1 * g) / 0.1 = g
    // v_hat = (0.001 * g^2) / 0.001 = g^2
    // Thus, weight update: w -= lr * g / (sqrt(g^2) + epsilon) ≈ w -= lr * sign(g)
    // Since g > 0, sign(g) = 1, so w -= lr * 1 = w - 0.1
    // Similarly for biases: b -= lr * sign(g) = b - 0.1 (for positive g) or b + 0.1 (for negative g)

    // Define expected updated weights and biases
    std::vector<std::vector<float>> expected_weights = {
        {0.9f, 1.9f}, // 1.0 - 0.1, 2.0 - 0.1
        {2.9f, 3.9f}  // 3.0 - 0.1, 4.0 - 0.1
    };

    std::vector<float> expected_biases = {0.4f, -0.4f}; // 0.5 - 0.1, -0.5 + 0.1

    // Verify updated weights
    bool weights_correct = true;
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            if (std::abs(weights[i][j] - expected_weights[i][j]) > 1e-5) {
                weights_correct = false;
                break;
            }
        }
        if (!weights_correct) break;
    }

    // Verify updated biases
    bool biases_correct = true;
    for (size_t i = 0; i < biases.size(); ++i) {
        if (std::abs(biases[i] - expected_biases[i]) > 1e-5) {
            biases_correct = false;
            break;
        }
    }

    // Output the test results
    if (weights_correct) {
        std::cout << "Weight updates are correct." << std::endl;
    } else {
        std::cout << "Weight updates are incorrect." << std::endl;
        std::cout << "Expected Weights:" << std::endl;
        for (const auto& row : expected_weights) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Actual Weights:" << std::endl;
        for (const auto& row : weights) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    if (biases_correct) {
        std::cout << "Bias updates are correct." << std::endl;
    } else {
        std::cout << "Bias updates are incorrect." << std::endl;
        std::cout << "Expected Biases:" << std::endl;
        for (const auto& val : expected_biases) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "Actual Biases:" << std::endl;
        for (const auto& val : biases) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    if (weights_correct && biases_correct) {
        std::cout << "Adam Optimizer Test Passed Successfully." << std::endl;
    } else {
        std::cout << "Adam Optimizer Test Failed." << std::endl;
    }

    return 0;
}
