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

// private:
    // Cached input
    std::vector<std::vector<float>> inputImage;

    // Function pointers for activation and its derivatives
    std::function<float(float)> act;
    std::function<float(float)> d_act;
    std::function<float(float)> d2_act;

    // Initialize activation functions and their derivatives
    void initializeFunctions();
};

#endif // ACTIVATION_H