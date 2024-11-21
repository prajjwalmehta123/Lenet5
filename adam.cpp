#include <iostream>
#include <cmath>
#include <vector>
#include <cassert> // For assert()
#include <omp.h>
#include "adam.h"

// Constructor
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(1) {}

// Update weights
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

    // Precompute constants
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float one_minus_beta1_t = 1.0f - beta1_t;
    float one_minus_beta2_t = 1.0f - beta2_t;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    // Flatten the nested loops
    int total_elements = rows * cols;

    // Parallelize updates with OpenMP
    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        int i = idx / cols;
        int j = idx % cols;

        // Update biased first and second moment estimates
        m_w[i][j] = beta1 * m_w[i][j] + one_minus_beta1 * gradients[i][j];
        v_w[i][j] = beta2 * v_w[i][j] + one_minus_beta2 * (gradients[i][j] * gradients[i][j]);

        // Compute bias-corrected first and second moment estimates
        float m_w_hat = m_w[i][j] / one_minus_beta1_t;
        float v_w_hat = v_w[i][j] / one_minus_beta2_t;

        // Update parameters
        weights[i][j] -= lr * m_w_hat / (std::sqrt(v_w_hat) + epsilon);
    }
}

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

    // Precompute constants
    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float one_minus_beta1_t = 1.0f - beta1_t;
    float one_minus_beta2_t = 1.0f - beta2_t;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    // Parallelize bias updates with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // Update biased first and second moments
        m_b[i] = beta1 * m_b[i] + one_minus_beta1 * gradients[i];
        v_b[i] = beta2 * v_b[i] + one_minus_beta2 * (gradients[i] * gradients[i]);

        // Correct bias
        float m_b_hat = m_b[i] / one_minus_beta1_t;
        float v_b_hat = v_b[i] / one_minus_beta2_t;

        // Update biases
        biases[i] -= lr * m_b_hat / (std::sqrt(v_b_hat) + epsilon);
    }
}