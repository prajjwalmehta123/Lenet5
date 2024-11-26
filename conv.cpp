#include "conv.h"
#include <cmath>
#include <random>

ConvolutionLayer::ConvolutionLayer(){}

ConvolutionLayer::ConvolutionLayer(ConvolutionLayer&& other) noexcept
    : inputChannels(other.inputChannels)
    , outputChannels(other.outputChannels)
    , kernelSize(other.kernelSize)
    , stride(other.stride)
    , padding(other.padding)
#ifdef USE_CUDA
    , gpuImplementation(std::move(other.gpuImplementation))
#else
    , weights(std::move(other.weights))
    , biases(std::move(other.biases))
    , gradWeights(std::move(other.gradWeights))
    , gradBiases(std::move(other.gradBiases))
    , inputDataBatch(std::move(other.inputDataBatch))
    , outputDataBatch(std::move(other.outputDataBatch))
    , weightOptimizers(std::move(other.weightOptimizers))
    , biasOptimizer(std::move(other.biasOptimizer))
#endif
{
}

ConvolutionLayer& ConvolutionLayer::operator=(ConvolutionLayer&& other) noexcept {
    if (this != &other) {
        inputChannels = other.inputChannels;
        outputChannels = other.outputChannels;
        kernelSize = other.kernelSize;
        stride = other.stride;
        padding = other.padding;
#ifdef USE_CUDA
        gpuImplementation = std::move(other.gpuImplementation);
#else
        weights = std::move(other.weights);
        biases = std::move(other.biases);
        gradWeights = std::move(other.gradWeights);
        gradBiases = std::move(other.gradBiases);
        inputDataBatch = std::move(other.inputDataBatch);
        outputDataBatch = std::move(other.outputDataBatch);
        weightOptimizers = std::move(other.weightOptimizers);
        biasOptimizer = std::move(other.biasOptimizer);
#endif
    }
    return *this;
}

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
    #ifdef USE_CUDA
    gpuImplementation = std::make_unique<ConvGPU>(inputChannels, outputChannels, kernelSize, stride, padding);
    #else
    weightOptimizers.resize(outputChannels, std::vector<AdamOptimizer>(inputChannels, AdamOptimizer(0.01,0.9,0.999,1e-8)));
    biasOptimizer = AdamOptimizer(0.01,0.9,0.999,1e-8);
    initializeWeights();
    #endif
}
ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::initializeWeights() {
    std::default_random_engine generator;
    //  He initialization for ReLU activation
    float stddev = std::sqrt(2.0f / (inputChannels * kernelSize * kernelSize));
    std::normal_distribution<float> distribution(0.0f, stddev);

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
    #ifdef USE_CUDA
    return gpuImplementation->forward(inputBatch, imageHeight, imageWidth);
#else
    inputDataBatch = inputBatch;
    outputDataBatch = conv2DCPU(inputBatch, imageHeight, imageWidth);
    return outputDataBatch;
#endif
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
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batchSize; ++b) {
        const std::vector<float>& inputFlat = inputBatch[b];
        std::vector<float>& outputFlat = outputBatch[b];

        // Create padded input
        std::vector<float> paddedInputFlat(paddedHeight * paddedWidth, 0.0f);
        for (int i = 0; i < imageHeight; ++i) {
            memcpy(&paddedInputFlat[(i + padding) * paddedWidth + padding],
                   &inputFlat[i * imageWidth],
                   sizeof(float) * imageWidth);
        }

        // Parallelize over output channels, height, and width
        #pragma omp parallel for collapse(3) schedule(static)
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    float sum = biases[oc];
                    int out_idx = oc * outputHeight * outputWidth + oh * outputWidth + ow;
                    #pragma omp simd reduction(+:sum)
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
                    outputFlat[out_idx] = sum;
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
    #ifdef USE_CUDA
    return gpuImplementation->backward(gradOutputBatch);
    #else
    size_t batchSize = gradOutputBatch.size();

    int inputSizePerImage = inputDataBatch[0].size(); // Total elements per image
    int inputHeight = static_cast<int>(std::sqrt(inputSizePerImage / inputChannels));
    int inputWidth = inputHeight;

    int outputHeight = calculateOutputSize(inputHeight, kernelSize, stride, padding);
    int outputWidth = calculateOutputSize(inputWidth, kernelSize, stride, padding);

    // Initialize gradients to zero
    gradWeights.assign(outputChannels, std::vector<std::vector<std::vector<float>>>(
        inputChannels, std::vector<std::vector<float>>(
            kernelSize, std::vector<float>(kernelSize, 0.0f))));
    gradBiases.assign(outputChannels, 0.0f); // Corrected line

    // Determine the number of threads
    int num_threads = omp_get_max_threads();

    // Create thread-local accumulators
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> gradWeightsLocal(
        num_threads, std::vector<std::vector<std::vector<std::vector<float>>>>(
            outputChannels, std::vector<std::vector<std::vector<float>>>(
                inputChannels, std::vector<std::vector<float>>(
                    kernelSize, std::vector<float>(kernelSize, 0.0f)))));

    std::vector<std::vector<float>> gradBiasesLocal(
        num_threads, std::vector<float>(outputChannels, 0.0f));

    // Initialize gradInputBatch to zeros
    std::vector<std::vector<float>> gradInputBatch(
        batchSize, std::vector<float>(inputDataBatch[0].size(), 0.0f));

    // Parallel processing
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static)
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
                        gradBiasesLocal[thread_id][oc] += gradOutputValue / batchSize;

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

                                        // Update gradWeightsLocal
                                        gradWeightsLocal[thread_id][oc][ic][kh][kw] += (gradOutputValue * inputValue) / batchSize;

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
    }

    // Aggregate gradBiasesLocal into gradBiases
    for (int t = 0; t < num_threads; ++t) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            gradBiases[oc] += gradBiasesLocal[t][oc];
        }
    }

    // Aggregate gradWeightsLocal into gradWeights
    for (int t = 0; t < num_threads; ++t) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int kh = 0; kh < kernelSize; ++kh) {
                    for (int kw = 0; kw < kernelSize; ++kw) {
                        gradWeights[oc][ic][kh][kw] += gradWeightsLocal[t][oc][ic][kh][kw];
                    }
                }
            }
        }
    }

    updateWeights();
    return gradInputBatch;
    #endif
}



void ConvolutionLayer::updateWeights() {
    #ifdef USE_CUDA
    gpuImplementation->updateWeights();
    #else
    // Update weights using AdamOptimizers
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            weightOptimizers[oc][ic].update_weight(weights[oc][ic], gradWeights[oc][ic]);
        }
    }
    biasOptimizer.update_bias(biases, gradBiases);
    #endif
}
std::vector<std::vector<std::vector<std::vector<float>>>> ConvolutionLayer::getWeights() const {
#ifdef USE_CUDA
    return gpuImplementation->getWeights();
#else
    return weights;
#endif
}

std::vector<float> ConvolutionLayer::getBiases() const {
#ifdef USE_CUDA
    return gpuImplementation->getBiases();
#else
    return biases;
#endif
}