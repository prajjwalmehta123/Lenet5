#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>


class Activation {
public:
    // Constructor to select the activation mode
    Activation();
    // Forward propagation
    std::vector<std::vector<float>> forwardProp(const std::vector<std::vector<float>>& input);

    // Backward propagation
    std::vector<std::vector<float>> backProp(const std::vector<std::vector<float>>& dZ);

private:
    // Cached input
    std::vector<std::vector<float>> inputImage;
    inline float relu(float x) { return x > 0 ? x : 0; }
    inline float d_relu(float x) { return x > 0 ? 1 : 0; }

    // Initialize activation functions and their derivatives
    void initializeFunctions();
};

#endif // ACTIVATION_H