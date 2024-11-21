#include <iostream>
#include <cmath>
#include <vector>
#include "adam.h"
// Update weights
#include <cassert> // For assert()


// Constructor
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}



void AdamOptimizer::update_weight(
    std::vector<std::vector<float>>& weights, 
    const std::vector<std::vector<float>>& gradients
) {
    int rows = weights.size();
    int cols = weights[0].size();

    // Initialize moments if not already done
    if (m_w.empty()) {
        m_w = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
        v_w = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
    }

    // Sanity check for moments
    assert(m_w.size() == rows && m_w[0].size() == cols);
    assert(v_w.size() == rows && v_w[0].size() == cols);

    t++; // Increment time step

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Update biased first and second moments
            m_w[i][j] = beta1 * m_w[i][j] + (1 - beta1) * gradients[i][j];
            v_w[i][j] = beta2 * v_w[i][j] + (1 - beta2) * (gradients[i][j] * gradients[i][j]);

            // Correct bias
            float m_w_hat = m_w[i][j] / (1 - std::pow(beta1, t));
            float v_w_hat = v_w[i][j] / (1 - std::pow(beta2, t));

            // Update weights
            weights[i][j] -= lr * m_w_hat / (std::sqrt(v_w_hat) + epsilon);
        }
    }
}

// Update biases
void AdamOptimizer::update_bias(
    std::vector<float>& biases, 
    const std::vector<float>& gradients
) {
    int size = biases.size();

    // Initialize moments if not already done
    if (m_b.empty()) {
        m_b = std::vector<float>(size, 0.0f);
        v_b = std::vector<float>(size, 0.0f);
    }

    for (int i = 0; i < size; ++i) {
        // Update biased first and second moments
        m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradients[i];
        v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradients[i] * gradients[i]);

        // Correct bias
        float m_b_hat = m_b[i] / (1 - std::pow(beta1, t));
        float v_b_hat = v_b[i] / (1 - std::pow(beta2, t));
        // Update biases
        biases[i] -= lr * m_b_hat / (std::sqrt(v_b_hat) + epsilon);
    }
}