#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::transform
#include"out.h"
#include <random>


OutputLayer::OutputLayer(){}
// Constructor to initialize weights and biases
OutputLayer::OutputLayer(int outputSize, int inputSize)
    : weights(outputSize, std::vector<float>(inputSize, 0.0f)), biases(outputSize, 0.0f) {
    initializeWeights();
    std::vector<std::vector<float>> weights; // Weight matrix [outputSize x inputSize]
    std::vector<float> biases;              // Bias vector [outputSize]
    std::vector<std::vector<float>> input;  // Cached input
    std::vector<float> output;              // Cached output
    std::vector<std::vector<float>> dWeights; // Gradients for weights
    std::vector<float> dBiases;              // Gradients for biases
    adam = AdamOptimizer(0.001,0.9,0.999,1e-8);
}

// Forward pass through the output layer
std::vector<std::vector<float>> OutputLayer::forwardProp(const std::vector<std::vector<float>>& input) {
    this->input = input; // Cache the input for backpropagation

    size_t batchSize = input.size();      // Number of samples in the batch
    size_t numOutputs = weights.size();   // Number of output neurons (e.g., 10 classes)
    size_t numInputs = weights[0].size(); // Number of input neurons (e.g., 84 from previous layer)

    std::vector<std::vector<float>> z(batchSize, std::vector<float>(numOutputs, 0.0f));
    std::vector<std::vector<float>> output(batchSize, std::vector<float>(numOutputs, 0.0f));

    // Compute z = W * input^T + b for each sample
    for (size_t sample = 0; sample < batchSize; ++sample) {
        for (size_t i = 0; i < numOutputs; ++i) {
            for (size_t j = 0; j < numInputs; ++j) {
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

    size_t batchSize = input.size();      // Number of samples
    size_t numOutputs = weights.size();   // Number of output neurons (10)
    size_t numInputs = weights[0].size(); // Number of input neurons (84)
    // Initialize gradients
    std::vector<std::vector<float>> dWeights(numOutputs, std::vector<float>(numInputs, 0.0f));
    std::vector<float> dBiases(numOutputs, 0.0f);

    // Gradient for input to propagate to the previous layer
    std::vector<std::vector<float>> dInput(batchSize, std::vector<float>(numInputs, 0.0f));

    // Compute gradients for weights, biases, and input for each sample
    for (size_t sample = 0; sample < batchSize; ++sample) {
        for (size_t i = 0; i < numOutputs; ++i) {
            dBiases[i] += dLoss[sample][i]; // Accumulate bias gradient
            for (size_t j = 0; j < numInputs; ++j) {
                dWeights[i][j] += dLoss[sample][i] * input[sample][j]; // Accumulate weight gradient
                dInput[sample][j] += dLoss[sample][i] * weights[i][j]; // Propagate gradient to input
            }
        }
    }

    // Average gradients over the batch
    for (size_t i = 0; i < numOutputs; ++i) {
        dBiases[i] /= static_cast<float>(batchSize);
        for (size_t j = 0; j < numInputs; ++j) {
            dWeights[i][j] /= static_cast<float>(batchSize);
        }
    }

    // Update weights and biases using Adam optimizer
    adam.update_weight(weights, dWeights);
    adam.update_bias(biases, dBiases);

    // Return gradient w.r.t. inputs to propagate to previous layer
    return dInput;
}


// Initialize weights with small random values

void OutputLayer::initializeWeights() {
    int n_in = weights.size();          // Number of input units
    int n_out = weights[0].size();      // Number of output units

    // Calculate the Xavier initialization range
    float limit = sqrt(6.0f / (n_in + n_out));

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Initialize weights with Xavier initialization
    for (auto& row : weights) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }

    // Initialize biases to zero
    std::fill(biases.begin(), biases.end(), 0.0f);
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