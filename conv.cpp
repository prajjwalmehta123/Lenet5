#include "conv.h"
#include <cmath>
#include <random>

ConvolutionLayer::ConvolutionLayer(){}

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
    weightOptimizers.resize(outputChannels, std::vector<AdamOptimizer>(inputChannels, AdamOptimizer(0.001,0.9,0.999,1e-8)));
    biasOptimizer = AdamOptimizer(0.001,0.9,0.999,1e-8);
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
    // gradOutputBatch: [batchSize][outputSizePerImage], where outputSizePerImage = outputChannels * outputHeight * outputWidth

    size_t batchSize = gradOutputBatch.size();

    int inputSizePerImage = inputDataBatch[0].size(); // Total elements per image
    int inputHeight = static_cast<int>(std::sqrt(inputSizePerImage / inputChannels));
    int inputWidth = inputHeight;

    int outputHeight = calculateOutputSize(inputHeight, kernelSize, stride, padding);
    int outputWidth = outputHeight;


    // Initialize gradients
    gradWeights = std::vector<std::vector<std::vector<std::vector<float>>>>(outputChannels,
        std::vector<std::vector<std::vector<float>>>(inputChannels,
            std::vector<std::vector<float>>(kernelSize, std::vector<float>(kernelSize, 0.0f))));
    gradBiases = std::vector<float>(outputChannels, 0.0f);

    // Initialize gradInputBatch to zeros
    std::vector<std::vector<float>> gradInputBatch(batchSize, std::vector<float>(inputDataBatch[0].size(), 0.0f));

    // Process each sample in the batch
    #pragma omp parallel for
    for (size_t b = 0; b < batchSize; ++b) {
        const std::vector<float>& inputFlat = inputDataBatch[b];        // [inputChannels * inputHeight * inputWidth]
        const std::vector<float>& gradOutputFlat = gradOutputBatch[b];  // [outputChannels * outputHeight * outputWidth]
        std::vector<float>& gradInputFlat = gradInputBatch[b];          // [inputChannels * inputHeight * inputWidth]

        // Loop over output channels
        for (int oc = 0; oc < outputChannels; ++oc) {
            // Loop over output spatial dimensions
            for (int h_out = 0; h_out < outputHeight; ++h_out) {
                for (int w_out = 0; w_out < outputWidth; ++w_out) {
                    int out_idx = oc * outputHeight * outputWidth + h_out * outputWidth + w_out;
                    float gradOutputValue = gradOutputFlat[out_idx];

                    // Update bias gradient
                    gradBiases[oc] += gradOutputValue / batchSize;  // Average over batch size

                    // Loop over input channels
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        // Loop over kernel dimensions
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int h_in = h_out * stride - padding + kh;
                                int w_in = w_out * stride - padding + kw;

                                // Check if indices are within bounds
                                if (h_in >= 0 && h_in < inputHeight && w_in >= 0 && w_in < inputWidth) {
                                    int in_idx = ic * inputHeight * inputWidth + h_in * inputWidth + w_in;
                                    float inputValue = inputFlat[in_idx];
                                    float weightValue = weights[oc][ic][kh][kw];

                                    // Update gradWeights
                                    gradWeights[oc][ic][kh][kw] += (gradOutputValue * inputValue) / batchSize;

                                    // Update gradInputFlat
                                    gradInputFlat[in_idx] += gradOutputValue * weightValue;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return gradInputBatch;
}


void ConvolutionLayer::updateWeights() {
    // Update weights using AdamOptimizers
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            weightOptimizers[oc][ic].update_weight(weights[oc][ic], gradWeights[oc][ic]);
        }
    }
    biasOptimizer.update_bias(biases, gradBiases);
}


void ConvolutionLayer::setGPUComputations(std::shared_ptr<GPUComputations> gpuComputations) {
    this->gpuComputations = gpuComputations;
}
