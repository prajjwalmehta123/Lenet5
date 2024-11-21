#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::transform
#include"out.h"


OutputLayer::OutputLayer(){}
// Constructor to initialize weights and biases
OutputLayer::OutputLayer(int inputSize, int outputSize)
    : weights(outputSize, std::vector<float>(inputSize)), biases(outputSize, 0.0f) {
    initializeWeights();
    std::vector<std::vector<float>> weights; // Weight matrix [outputSize x inputSize]
    std::vector<float> biases;              // Bias vector [outputSize]
    std::vector<std::vector<float>> input;               // Cached input
    std::vector<float> output;              // Cached output
    std::vector<std::vector<float>> dWeights; // Gradients for weights
    std::vector<float> dBiases;              // Gradients for biases
    adam = AdamOptimizer(0.001,0.9,0.999,1e-8);
}

// Forward pass through the output layer
std::vector<std::vector<float>> OutputLayer::forwardProp(const std::vector<std::vector<float>>& input) {
    this->input = input;
    size_t batchSize = input.size();      // Number of samples (60000)
    size_t numOutputs = weights[0].size();  // Number of output neurons (10)
    std::vector<std::vector<float>> z(batchSize, std::vector<float>(numOutputs, 0.0f));
    std::vector<std::vector<float>> output(batchSize, std::vector<float>(numOutputs, 0.0f));

    // Compute z = W * input + b for each sample
    for (size_t sample = 0; sample < batchSize; ++sample) {
        for (size_t i = 0; i < numOutputs; ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                z[sample][i] += weights[i][j] * input[sample][j];
            }
            z[sample][i] += biases[i]; // Add bias
        }
        output[sample] = softmax(z[sample]);
    }
    return output;
}

// Backward pass through the output layer
std::vector<std::vector<float>> OutputLayer::backProp(const std::vector<std::vector<float>>& dLoss) {

    size_t batchSize = input.size();      // Number of samples (60000)
    size_t numOutputs = weights[0].size();  // Number of output neurons (10)
    size_t numInputs = input[0].size();  // Number of input neurons (84)

    // Gradients for weights and biases
    std::vector<std::vector<float>> dWeights(numInputs, std::vector<float>(numOutputs, 0.0f));
    std::vector<float> dBiases(numOutputs, 0.0f);

    // Gradient for input to propagate to the previous layer
    std::vector<std::vector<float>> dInput(batchSize, std::vector<float>(numInputs, 0.0f));

    // Compute gradients for weights, biases, and input for each sample
    for (size_t sample = 0; sample < batchSize; ++sample) {
        for (size_t i = 0; i < numInputs; ++i) {
            dBiases[i] += dLoss[sample][i]; // Accumulate bias gradient
            for (size_t j = 0; j < numOutputs; ++j) {
                dWeights[i][j] += dLoss[sample][i] * input[sample][j]; // Accumulate weight gradient
                dInput[sample][j] += dLoss[sample][i] * weights[i][j]; // Propagate gradient to input
            }
        }
    }

    // Average gradients over the batch
    for (size_t i = 0; i < numOutputs; ++i) {
        dBiases[i] /= batchSize;
        for (size_t j = 0; j < numInputs; ++j) {
            dWeights[i][j] /= batchSize;
        }
    }

    adam.update_weight(weights, dWeights);
    adam.update_bias(biases, dBiases);

    // Debugging: Print updated weights and biases
    std::cout << "Updated weights and biases.\n";
    return dInput;
}


// Initialize weights with small random values
void OutputLayer::initializeWeights() {
    for (auto& row : weights) {
        for (auto& val : row) {
            val = static_cast<float>(rand()) / RAND_MAX * 0.01f; // Small random values
        }
    }
}

// Softmax activation function
std::vector<float> OutputLayer::softmax(const std::vector<float>& z) {
    std::vector<float> expZ(z.size());
    float maxZ = *std::max_element(z.begin(), z.end()); // Prevent overflow
    float sumExpZ = 0.0f;

    for (size_t i = 0; i < z.size(); ++i) {
        expZ[i] = std::exp(z[i] - maxZ); // Shift for numerical stability
        sumExpZ += expZ[i];
    }

    for (size_t i = 0; i < expZ.size(); ++i) {
        expZ[i] /= sumExpZ; // Normalize to probabilities
    }
    return expZ;
}