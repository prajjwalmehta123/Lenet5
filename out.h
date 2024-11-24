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
    OutputLayer(int outputSize, int inputSize);

    // Forward pass through the output layer
    std::vector<std::vector<float>> forwardProp(const std::vector<std::vector<float>>& input);

    // Backward pass through the output layer
    std::vector<std::vector<float>> backProp(const std::vector<std::vector<float>>& dLoss);

private:

    std::vector<std::vector<float>> weights; // 2D vector for weights
    std::vector<float> biases;              // 1D vector for biases
    std::vector<std::vector<float>> input;  // Cached input
    AdamOptimizer adam;                     // Adam optimizer instance

    size_t numOutputs;                      // Number of outputs
    size_t numInputs;                       // Number of inputs

    std::vector<std::vector<float>> weightsTransposed; // Pre-transposed weights for efficiency

    void initializeWeights();
    void transposeWeights();
    std::vector<float> softmax(const std::vector<float>& z);
};

#endif // OUT_H