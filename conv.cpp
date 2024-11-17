#include "conv.h"
#include <cmath>
#include <random>

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
    initializeWeights();
}

void ConvolutionLayer::initializeWeights() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.05f);

    weights.resize(outputChannels, std::vector<std::vector<std::vector<float>>>(inputChannels,
                    std::vector<std::vector<float>>(kernelSize, std::vector<float>(kernelSize))));
    biases.resize(outputChannels, 0.0f);

    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            for (int kh = 0; kh < kernelSize; ++kh) {
                for (int kw = 0; kw < kernelSize; ++kw) {
                    weights[oc][ic][kh][kw] = distribution(generator);
                }
            }
        }
        biases[oc] = 0.0f;
    }
}

std::vector<std::vector<float>> ConvolutionLayer::forward(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth) {
    inputDataBatch = inputBatch; // Save input for backward pass

    if (gpuComputations) {
        std::cout << "GPU computations not yet implemented." << std::endl;
        return {};
    } else {
        // CPU computations
        outputDataBatch = conv2DCPU(inputBatch, imageHeight, imageWidth);
        return outputDataBatch;
    }
}

// CPU convolution for a batch with flattened images
std::vector<std::vector<float>> ConvolutionLayer::conv2DCPU(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth) {
    size_t batchSize = inputBatch.size();

    int paddedHeight = imageHeight + 2 * padding;
    int paddedWidth = imageWidth + 2 * padding;

    int outputHeight = calculateOutputSize(imageHeight, kernelSize, stride, padding);
    int outputWidth = calculateOutputSize(imageWidth, kernelSize, stride, padding);
    int outputSizePerImage = outputHeight * outputWidth * outputChannels;
    std::vector<std::vector<float>> outputBatch(batchSize, std::vector<float>(outputSizePerImage, 0.0f));

    // Parallelize over the batch dimension
    #pragma omp parallel for
    for (size_t b = 0; b < batchSize; ++b) {
        const std::vector<float>& inputFlat = inputBatch[b];
        std::vector<float>& outputFlat = outputBatch[b];

        int paddedSize = paddedHeight * paddedWidth;
        std::vector<float> paddedInputFlat(paddedSize, 0.0f);
        for (int i = 0; i < imageHeight; ++i) {
            for (int j = 0; j < imageWidth; ++j) {
                int inputIndex = i * imageWidth + j;
                int paddedIndex = (i + padding) * paddedWidth + (j + padding);
                paddedInputFlat[paddedIndex] = inputFlat[inputIndex];
            }
        }
        // Perform convolution
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    float sum = biases[oc];
                    int outputIndex = oc * outputHeight * outputWidth + oh * outputWidth + ow;
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                int paddedIndex = ih * paddedWidth + iw;
                                sum += weights[oc][ic][kh][kw] * paddedInputFlat[paddedIndex];
                            }
                        }
                    }
                    outputFlat[outputIndex] = sum;
                }
            }
        }
    }
    return outputBatch;
}

int ConvolutionLayer::calculateOutputSize(int inputSize, int kernelSize, int stride, int padding) {
    return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
}
std::vector<std::vector<float>> ConvolutionLayer::backward(const std::vector<std::vector<float>>& gradOutputBatch) {
    // To be implemented
    std::cout << "Backward pass not yet implemented." << std::endl;
    return {};
}


void ConvolutionLayer::updateWeights(float learningRate) {
    // To be implemented
    std::cout << "Weight update not yet implemented." << std::endl;
}

void ConvolutionLayer::setGPUComputations(std::shared_ptr<GPUComputations> gpuComputations) {
    this->gpuComputations = gpuComputations;
}
