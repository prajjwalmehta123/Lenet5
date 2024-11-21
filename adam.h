#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include <vector>
#include <cmath>

class AdamOptimizer {
public:
    AdamOptimizer();
    // Constructor to initialize Adam optimizer parameters
    // AdamOptimizer(float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);

    // Update function for weights
    void update_weight(
        std::vector<std::vector<float>>& weights, 
        const std::vector<std::vector<float>>& gradients
    );

    // Update function for biases
    void update_bias(
        std::vector<float>& biases, 
        const std::vector<float>& gradients
    );

private:
    float lr;         // Learning rate
    float beta1;      // Decay rate for the first moment
    float beta2;      // Decay rate for the second moment
    float epsilon;    // Small constant to prevent division by zero

    int t;            // Time step

    // Moment vectors for weights
    std::vector<std::vector<float>> m_w;  // First moment
    std::vector<std::vector<float>> v_w;  // Second moment

    // Moment vectors for biases
    std::vector<float> m_b;  // First moment
    std::vector<float> v_b;  // Second moment
};

#endif // ADAMOPTIMIZER_H