#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include "activation.h"

std::vector<std::vector<float>> inputImage; // Cached input

// Activation function pointers
std::function<float(float)> act;
std::function<float(float)> d_act;
std::function<float(float)> d2_act;

    // Initialize functions and derivatives
void Activation::initializeFunctions() {
    // ReLU
    this->act = [](float x) { return x > 0 ? x : 0; };
    this->d_act = [](float x) { return x > 0 ? 1 : 0; };
    this->d2_act = [](float x) { return 0; };
}

// Constructor to initialize the activation mode
Activation::Activation() {
    initializeFunctions();
}

// Forward propagation
std::vector<std::vector<float>> Activation::forwardProp(const std::vector<std::vector<float>>& input) {
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
std::vector<std::vector<float>> Activation::backProp(const std::vector<std::vector<float>>& dZ) {
    std::vector<std::vector<float>> dA(dZ.size(), std::vector<float>(dZ[0].size()));

    for (size_t i = 0; i < dZ.size(); ++i) {
        for (size_t j = 0; j < dZ[i].size(); ++j) {
            dA[i][j] = dZ[i][j] * d_act(inputImage[i][j]);
        }
    }
    return dA;
}