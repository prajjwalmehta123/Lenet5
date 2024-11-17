#include "conv.h"

#include <random>

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
    initializeWeights();
}

// Initialize weights and biases with random values
void ConvolutionLayer::initializeWeights() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.05f);

    weights.resize(outputChannels, std::vector<std::vector<std::vector<float>>>(inputChannels,
                std::vector<std::vector<float>>(kernelSize, std::vector<float>(kernelSize))));
    biases.resize(outputChannels, 0.0f);

    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            for (int i = 0; i < kernelSize; ++i) {
                for (int j = 0; j < kernelSize; ++j) {
                    weights[oc][ic][i][j] = distribution(generator);
                }
            }
        }
        biases[oc] = 0.0f;
    }
}

std::vector<std::vector<float>> ConvolutionLayer::forward(const std::vector<std::vector<float>>& input) {
    inputData = input;
    if (gpuComputations) {
        std::cout << "GPU computations not yet implemented." << std::endl;
        // Implement GPU forward pass here
    } else {
        outputData = conv2DCPU(input);
    }
    return outputData;
}

std::vector<std::vector<float>> ConvolutionLayer::conv2DCPU(const std::vector<std::vector<float>>& input) {

}
