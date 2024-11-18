#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>
#include <string>
#include <utility>
#include <random>

class FCLayer {
private:
    std::vector<std::vector<float>> weight; // Weight matrix
    std::vector<float> bias;               // Bias vector
    std::vector<std::vector<float>> v_w;   // Velocity for weights (momentum)
    std::vector<float> v_b;                // Velocity for biases (momentum)
    std::vector<std::vector<float>> input_array; // Cached input for backpropagation
    float lr;                              // Learning rate

    // Helper function to initialize weights and biases
    std::pair<std::vector<std::vector<float>>, std::vector<float>> initialize(
        int rows, int cols, const std::string& mode);

    // Helper function to update weights and biases
    void update(
        std::vector<std::vector<float>>& weight, std::vector<float>& bias,
        const std::vector<std::vector<float>>& dW, const std::vector<float>& db,
        std::vector<std::vector<float>>& v_w, std::vector<float>& v_b,
        float lr, float momentum, float weight_decay);

public:
    FCLayer();
    // Constructor: Initialize layer with weight dimensions and initialization mode
    FCLayer(const std::pair<int, int>& weight_shape, const std::string& init_mode = "Gaussian_dist");

    // Forward Propagation: Compute output for given input
    std::vector<std::vector<float>> forward_prop(const std::vector<std::vector<float>>& input_array);

    // Backward Propagation: Compute gradients and update weights
    std::vector<std::vector<float>> back_prop(const std::vector<std::vector<float>>& dZ, float momentum, float weight_decay);

    // Stochastic Diagonal Levenberg-Marquardt (SDLM): Compute Hessian approximation
    std::vector<std::vector<float>> SDLM(const std::vector<std::vector<float>>& d2Z, float mu, float lr_global);
};

#endif // FCLAYER_H