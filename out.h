#ifndef OUT_H
#define OUT_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "adam.h"

class OutputLayer {
public:
    OutputLayer();
    // Constructor to initialize weights and biases
    OutputLayer(int inputSize, int outputSize);
    AdamOptimizer adam;

    // Forward pass through the output layer
    std::vector<std::vector<float>> forwardProp(const std::vector<std::vector<float>>& input);

    // Backward pass through the output layer
    std::vector<std::vector<float>> backProp(const std::vector<std::vector<float>>& dLoss);

    // Update weights and biases
    void updateWeights(float learningRate);
    // Initialize weights with small random values
    void initializeWeights();

    // Softmax activation function
    std::vector<float> softmax(const std::vector<float>& z);

private:
    std::vector<std::vector<float>> weights; // Weight matrix [outputSize x inputSize]
    std::vector<float> biases;              // Bias vector [outputSize]
    std::vector<std::vector<float>> input;               // Cached input
    std::vector<float> output;              // Cached output
    std::vector<std::vector<float>> dWeights; // Gradients for weights
    std::vector<float> dBiases;              // Gradients for biases

    
};

#endif // OUT_H