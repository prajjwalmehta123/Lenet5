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

    // Debugging: Check weights and gradients dimensions
    std::cout << "Updating weights...\n";
    std::cout << "Weights dimensions: " << rows << " x " << cols << "\n";
    std::cout << "Gradients dimensions: " << gradients.size() 
              << " x " << (gradients.empty() ? 0 : gradients[0].size()) << "\n";

    // Check dimension match
    if (gradients.size() != rows || gradients[0].size() != cols) {
        std::cerr << "Error: Gradients dimensions do not match weights dimensions.\n";
        return;
    }

    // Initialize moments if not already done
    if (m_w.empty()) {
        std::cout << "Initializing moment vectors for weights...\n";
        m_w = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
        v_w = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
    }

    // Sanity check for moments
    assert(m_w.size() == rows && m_w[0].size() == cols);
    assert(v_w.size() == rows && v_w[0].size() == cols);

    t++; // Increment time step
    std::cout << "Time step t: " << t << "\n";

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Validate indices before access
            assert(i < m_w.size() && j < m_w[0].size());
            assert(i < v_w.size() && j < v_w[0].size());
            assert(i < gradients.size() && j < gradients[i].size());
            assert(i < weights.size() && j < weights[i].size());

            // Update biased first and second moments
            m_w[i][j] = beta1 * m_w[i][j] + (1 - beta1) * gradients[i][j];
            v_w[i][j] = beta2 * v_w[i][j] + (1 - beta2) * (gradients[i][j] * gradients[i][j]);

            // Debugging: Print intermediate values
            std::cout << "Weight [" << i << "][" << j << "] - Gradient: " << gradients[i][j] 
                      << ", m_w: " << m_w[i][j] << ", v_w: " << v_w[i][j] << "\n";

            // Correct bias
            float m_w_hat = m_w[i][j] / (1 - std::pow(beta1, t));
            float v_w_hat = v_w[i][j] / (1 - std::pow(beta2, t));

            // Debugging: Print corrected moments
            std::cout << "m_w_hat: " << m_w_hat << ", v_w_hat: " << v_w_hat << "\n";

            // Update weights
            weights[i][j] -= lr * m_w_hat / (std::sqrt(v_w_hat) + epsilon);

            // Debugging: Print updated weight
            std::cout << "Updated weight [" << i << "][" << j << "]: " << weights[i][j] << "\n";
        }
    }
}

// Update biases
void AdamOptimizer::update_bias(
    std::vector<float>& biases, 
    const std::vector<float>& gradients
) {
    int size = biases.size();

    // Debugging: Check biases and gradients dimensions
    std::cout << "Updating biases...\n";
    std::cout << "Biases size: " << size << "\n";
    std::cout << "Gradients size: " << gradients.size() << "\n";

    if (gradients.size() != size) {
        std::cerr << "Error: Gradients size does not match biases size.\n";
        return;
    }

    // Initialize moments if not already done
    if (m_b.empty()) {
        std::cout << "Initializing moment vectors for biases...\n";
        m_b = std::vector<float>(size, 0.0f);
        v_b = std::vector<float>(size, 0.0f);
    }

    for (int i = 0; i < size; ++i) {
        // Update biased first and second moments
        m_b[i] = beta1 * m_b[i] + (1 - beta1) * gradients[i];
        v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gradients[i] * gradients[i]);

        // Debugging: Print intermediate values
        std::cout << "Bias [" << i << "] - Gradient: " << gradients[i] 
                  << ", m_b: " << m_b[i] << ", v_b: " << v_b[i] << "\n";

        // Correct bias
        float m_b_hat = m_b[i] / (1 - std::pow(beta1, t));
        float v_b_hat = v_b[i] / (1 - std::pow(beta2, t));

        // Debugging: Print corrected moments
        std::cout << "m_b_hat: " << m_b_hat << ", v_b_hat: " << v_b_hat << "\n";

        // Update biases
        biases[i] -= lr * m_b_hat / (std::sqrt(v_b_hat) + epsilon);

        // Debugging: Print updated bias
        std::cout << "Updated bias [" << i << "]: " << biases[i] << "\n";
    }
}