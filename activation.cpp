#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

class Activation {
public:
    // Constructor to initialize the activation mode
    Activation() {
        initializeFunctions();
    }

    // Forward propagation
    std::vector<std::vector<float>> forwardProp(const std::vector<std::vector<float>>& input) {
        inputImage = input; // Cache input for backpropagation
        std::vector<std::vector<float>> output(input.size(), std::vector<float>(input[0].size()));

        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < input[i].size(); ++j) {
                output[i][j] = act(input[i][j]);
            }
        }
        return output;
    }

    // Backward propagation
    std::vector<std::vector<float>> backProp(const std::vector<std::vector<float>>& dZ) {
        std::vector<std::vector<float>> dA(dZ.size(), std::vector<float>(dZ[0].size()));

        for (size_t i = 0; i < dZ.size(); ++i) {
            for (size_t j = 0; j < dZ[i].size(); ++j) {
                dA[i][j] = dZ[i][j] * d_act(inputImage[i][j]);
            }
        }
        return dA;
    }

    // Stochastic Diagonal Levenberg-Marquardt
    std::vector<std::vector<float>> SDLM(const std::vector<std::vector<float>>& d2Z) {
        std::vector<std::vector<float>> dA(d2Z.size(), std::vector<float>(d2Z[0].size()));

        for (size_t i = 0; i < d2Z.size(); ++i) {
            for (size_t j = 0; j < d2Z[i].size(); ++j) {
                float derivative = d_act(inputImage[i][j]);
                dA[i][j] = d2Z[i][j] * derivative * derivative;
            }
        }
        return dA;
    }

private:
    std::vector<std::vector<float>> inputImage; // Cached input

    // Activation function pointers
    std::function<float(float)> act;
    std::function<float(float)> d_act;
    std::function<float(float)> d2_act;

    // Initialize functions and derivatives
    void initializeFunctions() {
        // ReLU
        this->act = [](float x) { return x > 0 ? x : 0; };
        this->d_act = [](float x) { return x > 0 ? 1 : 0; };
        this->d2_act = [](float x) { return 0; };
    }
};