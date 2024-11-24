#include "conv.h"
#include <cmath>
#include <random>

ConvolutionLayer::ConvolutionLayer(){}

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize), stride(stride), padding(padding) {
    weightOptimizers.resize(outputChannels, std::vector<AdamOptimizer>(inputChannels, AdamOptimizer(0.01,0.9,0.999,1e-8)));
    biasOptimizer = AdamOptimizer(0.01,0.9,0.999,1e-8);
    initializeWeights();
}

void ConvolutionLayer::initializeWeights() {
    std::default_random_engine generator;
    float stddev = std::sqrt(2.0f / (inputChannels * kernelSize * kernelSize));
    std::normal_distribution<float> distribution(0.0f, stddev);

    // Calculate total number of weights
    const int total_weights = outputChannels * inputChannels * kernelSize * kernelSize;

    // Resize with zero initialization
    weights_flat.clear();
    weights_flat.resize(total_weights, 0.0f);
    gradWeights_flat.resize(total_weights, 0.0f);

    // Initialize weights
    for (int i = 0; i < total_weights; ++i) {
        weights_flat[i] = distribution(generator);
    }

    // Initialize biases
    biases.clear();
    biases.resize(outputChannels, 0.0f);
}

std::vector<std::vector<float>> ConvolutionLayer::forward(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth) {
    inputDataBatch = inputBatch; // Save input for backward pass
    outputDataBatch = conv2DCPU(inputBatch, imageHeight, imageWidth);
    return outputDataBatch;
}

std::vector<std::vector<float>> ConvolutionLayer::conv2DCPU(
    const std::vector<std::vector<float>>& inputBatch,
    int imageHeight,
    int imageWidth) {

    validateDimensions(inputBatch, imageHeight, imageWidth);

    const size_t batchSize = inputBatch.size();
    const int outputHeight = calculateOutputSize(imageHeight, kernelSize, stride, padding);
    const int outputWidth = calculateOutputSize(imageWidth, kernelSize, stride, padding);
    const int outputChannelSize = outputHeight * outputWidth;
    const int outputTotalSize = outputChannelSize * outputChannels;

    // Initialize output batch
    std::vector<std::vector<float>> outputBatch(batchSize,
        std::vector<float>(outputTotalSize, 0.0f));

    // Process each sample in the batch
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batchSize; ++b) {
        // For each output channel
        for (int oc = 0; oc < outputChannels; ++oc) {
            // For each output position
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    float sum = biases[oc];

                    // For each input channel
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        // For each kernel position
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                // Skip if outside input bounds
                                if (ih >= 0 && ih < imageHeight && iw >= 0 && iw < imageWidth) {
                                    // Calculate indices carefully
                                    int input_idx = (ic * imageHeight * imageWidth) + (ih * imageWidth) + iw;
                                    int weight_idx = ((oc * inputChannels + ic) * kernelSize + kh) * kernelSize + kw;

                                    // Bounds checking
                                    if (input_idx >= 0 && input_idx < inputBatch[b].size() &&
                                        weight_idx >= 0 && weight_idx < weights_flat.size()) {
                                        sum += inputBatch[b][input_idx] * weights_flat[weight_idx];
                                    }
                                }
                            }
                        }
                    }

                    // Calculate output index
                    int out_idx = (oc * outputHeight * outputWidth) + (oh * outputWidth) + ow;
                    if (out_idx >= 0 && out_idx < outputBatch[b].size()) {
                        outputBatch[b][out_idx] = sum;
                    }
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
    size_t batchSize = gradOutputBatch.size();
    int inputSizePerImage = inputDataBatch[0].size();
    int inputHeight = static_cast<int>(std::sqrt(inputSizePerImage / inputChannels));
    int inputWidth = inputHeight;

    // Calculate output dimensions
    int outputHeight = calculateOutputSize(inputHeight, kernelSize, stride, padding);
    int outputWidth = calculateOutputSize(inputWidth, kernelSize, stride, padding);
    const int outputSizePerChannel = outputHeight * outputWidth;

    // Initialize gradients
    gradWeights_flat.assign(weights_flat.size(), 0.0f);
    gradBiases.assign(outputChannels, 0.0f);

    // Thread-local storage
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> gradWeights_local(num_threads,
        std::vector<float>(weights_flat.size(), 0.0f));
    std::vector<std::vector<float>> gradBiases_local(num_threads,
        std::vector<float>(outputChannels, 0.0f));

    // Initialize gradient input
    std::vector<std::vector<float>> gradInputBatch(
        batchSize, std::vector<float>(inputSizePerImage, 0.0f));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (size_t b = 0; b < batchSize; ++b) {
            const std::vector<float>& inputFlat = inputDataBatch[b];
            const std::vector<float>& gradOutputFlat = gradOutputBatch[b];
            std::vector<float>& gradInputFlat = gradInputBatch[b];

            // Process each output channel
            for (int oc = 0; oc < outputChannels; ++oc) {
                const int outputChannelOffset = oc * outputSizePerChannel;

                // Process each output position
                for (int h_out = 0; h_out < outputHeight; ++h_out) {
                    for (int w_out = 0; w_out < outputWidth; ++w_out) {
                        // Calculate output gradient index
                        const int out_idx = outputChannelOffset + h_out * outputWidth + w_out;
                        const float gradOutputValue = gradOutputFlat[out_idx];

                        // Update bias gradient (thread-local)
                        gradBiases_local[thread_id][oc] += gradOutputValue / batchSize;

                        // Process each input channel
                        for (int ic = 0; ic < inputChannels; ++ic) {
                            const int inputChannelOffset = ic * inputHeight * inputWidth;

                            // Process each kernel position
                            for (int kh = 0; kh < kernelSize; ++kh) {
                                const int h_in = h_out * stride - padding + kh;

                                if (h_in >= 0 && h_in < inputHeight) {
                                    for (int kw = 0; kw < kernelSize; ++kw) {
                                        const int w_in = w_out * stride - padding + kw;

                                        if (w_in >= 0 && w_in < inputWidth) {
                                            // Calculate input index
                                            const int in_idx = inputChannelOffset + h_in * inputWidth + w_in;

                                            // Calculate weight index
                                            const int weight_idx = get_weight_index(oc, ic, kh, kw);

                                            if (weight_idx < weights_flat.size() &&
                                                in_idx < inputFlat.size()) {

                                                const float inputValue = inputFlat[in_idx];
                                                const float weightValue = weights_flat[weight_idx];

                                                // Update weight gradients (thread-local)
                                                gradWeights_local[thread_id][weight_idx] +=
                                                    (gradOutputValue * inputValue) / batchSize;

                                                // Update input gradients
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
        }
    }

    // Aggregate bias gradients
    for (int t = 0; t < num_threads; ++t) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            gradBiases[oc] += gradBiases_local[t][oc];
        }
    }

    // Aggregate weight gradients
    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < weights_flat.size(); ++i) {
            gradWeights_flat[i] += gradWeights_local[t][i];
        }
    }

    updateWeights();
    return gradInputBatch;
}



void ConvolutionLayer::updateWeights() {
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            // Update each weight using the flattened arrays
            for (int kh = 0; kh < kernelSize; ++kh) {
                for (int kw = 0; kw < kernelSize; ++kw) {
                    const int idx = get_weight_index(oc, ic, kh, kw);
                    weightOptimizers[oc][ic].update_single_weight(
                        weights_flat[idx],
                        gradWeights_flat[idx]
                    );
                }
            }
        }
    }
    biasOptimizer.update_bias(biases, gradBiases);
}

