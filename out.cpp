#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::transform
#include <random>
#include <omp.h> // For parallelization
#include "out.h"

OutputLayer::OutputLayer(){};


OutputLayer::OutputLayer(int outputSize, int inputSize)
    : weights(outputSize, std::vector<float>(inputSize, 0.0f)),
      biases(outputSize, 0.0f), numOutputs(outputSize), numInputs(inputSize) {
    #ifdef USE_CUDA
    gpuImplementation = std::make_unique<OutputLayerGPU>(outputSize, inputSize);
    #endif
    initializeWeights();
    adam = AdamOptimizer(0.01, 0.9, 0.999, 1e-8);
    transposeWeights(); // Precompute the transposed weights
}
OutputLayer::OutputLayer(OutputLayer&& other) noexcept 
#ifdef USE_CUDA
    : gpuImplementation(std::move(other.gpuImplementation))
#else
    : weights(std::move(other.weights))
    , biases(std::move(other.biases))
    , input(std::move(other.input))
    , adam(std::move(other.adam))
    , numOutputs(other.numOutputs)
    , numInputs(other.numInputs)
    , weightsTransposed(std::move(other.weightsTransposed))
#endif
{
}

// Move assignment operator
OutputLayer& OutputLayer::operator=(OutputLayer&& other) noexcept {
    if (this != &other) {
#ifdef USE_CUDA
        gpuImplementation = std::move(other.gpuImplementation);
#else
        weights = std::move(other.weights);
        biases = std::move(other.biases);
        input = std::move(other.input);
        adam = std::move(other.adam);
        numOutputs = other.numOutputs;
        numInputs = other.numInputs;
        weightsTransposed = std::move(other.weightsTransposed);
#endif
    }
    return *this;
}

// Transpose weights for cache-friendly access in backprop
void OutputLayer::transposeWeights() {
    weightsTransposed.resize(numInputs, std::vector<float>(numOutputs));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numOutputs; ++i) {
        for (size_t j = 0; j < numInputs; ++j) {
            weightsTransposed[j][i] = weights[i][j];
        }
    }
}

// Forward Propagation
std::vector<std::vector<float>> OutputLayer::forwardProp(const std::vector<std::vector<float>>& input) {
    #ifdef USE_CUDA
    return gpuImplementation->forward(input);
    #endif
    this->input = input; // Cache input
    size_t batchSize = input.size();
    std::vector<std::vector<float>> z(batchSize, std::vector<float>(numOutputs, 0.0f));
    std::vector<std::vector<float>> output(batchSize, std::vector<float>(numOutputs));

    // Compute z = W * input + b
    #pragma omp parallel for collapse(2)
    for (size_t sample = 0; sample < batchSize; ++sample) {
        for (size_t i = 0; i < numOutputs; ++i) {
            float sum = biases[i];
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < numInputs; ++j) {
                sum += weights[i][j] * input[sample][j];
            }
            z[sample][i] = sum;
        }
    }

    // Apply softmax
    #pragma omp parallel for
    for (size_t sample = 0; sample < batchSize; ++sample) {
        output[sample] = softmax(z[sample]);
    }

    return output;
}

// Backward Propagation
std::vector<std::vector<float>> OutputLayer::backProp(const std::vector<std::vector<float>>& dLoss) {
    #ifdef USE_CUDA
    return gpuImplementation->backward(dLoss);
    #endif
    size_t batchSize = input.size();
    std::vector<std::vector<float>> dWeights(numOutputs, std::vector<float>(numInputs, 0.0f));
    std::vector<float> dBiases(numOutputs, 0.0f);
    std::vector<std::vector<float>> dInput(batchSize, std::vector<float>(numInputs, 0.0f));

    // Compute gradients using transposed weights
    #pragma omp parallel
    {
        std::vector<std::vector<float>> local_dWeights(numOutputs, std::vector<float>(numInputs, 0.0f));
        std::vector<float> local_dBiases(numOutputs, 0.0f);

        #pragma omp for collapse(2)
        for (size_t sample = 0; sample < batchSize; ++sample) {
            for (size_t i = 0; i < numOutputs; ++i) {
                local_dBiases[i] += dLoss[sample][i];
                #pragma omp simd
                for (size_t j = 0; j < numInputs; ++j) {
                    local_dWeights[i][j] += dLoss[sample][i] * input[sample][j];
                    dInput[sample][j] += dLoss[sample][i] * weightsTransposed[j][i];
                }
            }
        }

        // Merge local gradients into global buffers
        #pragma omp critical
        {
            for (size_t i = 0; i < numOutputs; ++i) {
                dBiases[i] += local_dBiases[i];
                for (size_t j = 0; j < numInputs; ++j) {
                    dWeights[i][j] += local_dWeights[i][j];
                }
            }
        }
    }

    // Average gradients
    float scale = 1.0f / batchSize;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numOutputs; ++i) {
        for (size_t j = 0; j < numInputs; ++j) {
            dWeights[i][j] *= scale;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < numOutputs; ++i) {
        dBiases[i] *= scale;
    }

    // Update weights and biases
    adam.update_weight(weights, dWeights);
    adam.update_bias(biases, dBiases);

    return dInput;
}

// Initialize weights using Xavier initialization
void OutputLayer::initializeWeights() {
    float limit = sqrt(6.0f / (numInputs + numOutputs));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numOutputs; ++i) {
        for (size_t j = 0; j < numInputs; ++j) {
            weights[i][j] = dist(gen);
        }
    }
    std::fill(biases.begin(), biases.end(), 0.0f);
}

// Softmax activation function
std::vector<float> OutputLayer::softmax(const std::vector<float>& z) {
    std::vector<float> expZ(z.size());
    float maxZ = *std::max_element(z.begin(), z.end()); // Prevent overflow
    float sumExpZ = 0.0f;

    #pragma omp parallel for reduction(+:sumExpZ)
    for (size_t i = 0; i < z.size(); ++i) {
        expZ[i] = std::exp(z[i] - maxZ);
        sumExpZ += expZ[i];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < expZ.size(); ++i) {
        expZ[i] /= sumExpZ;
    }
    return expZ;
}
std::vector<std::vector<float>> OutputLayer::getWeights() const {
#ifdef USE_CUDA
    return gpuImplementation->getWeights();
#else
    return weights;
#endif
}

std::vector<float> OutputLayer::getBiases() const {
#ifdef USE_CUDA
    return gpuImplementation->getBiases();
#else
    return biases;
#endif
}