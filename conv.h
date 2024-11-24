// conv.h

#ifndef CONV_H
#define CONV_H

#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include "adam.h"
#include <omp.h>


class ConvGPU;

class ConvolutionLayer {
public:
    // Constructors
    ConvolutionLayer();
    ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int padding = 0);

    // Forward and Backward Passes
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutputBatch);
    void updateWeights();
    static int calculateOutputSize(int inputSize, int kernelSize, int stride, int padding);

private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [outChannels][inChannels][kernelSize][kernelSize]
    std::vector<float> biases; // [outChannels]
    std::vector<std::vector<std::vector<std::vector<float>>>> gradWeights; // [outChannels][inChannels][kernelSize][kernelSize]
    std::vector<float> gradBiases; // [outChannels]
    std::vector<std::vector<float>> inputDataBatch; // Flattened input batch
    std::vector<std::vector<float>> outputDataBatch; // Flattened output batch
    std::vector<std::vector<float>> conv2DCPU(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth);
    void initializeWeights();
    std::vector<std::vector<AdamOptimizer>> weightOptimizers;
    AdamOptimizer biasOptimizer;
};

#endif // CONV_H
