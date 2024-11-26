// conv.h
#ifndef CONV_H
#define CONV_H

#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include "adam.h"
#include <omp.h>

#ifdef USE_CUDA
#include "conv_gpu.cuh"
#endif

class ConvolutionLayer {
public:
    ConvolutionLayer();
    ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int padding = 0);
    ~ConvolutionLayer();
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutputBatch);
    void updateWeights();
    static int calculateOutputSize(int inputSize, int kernelSize, int stride, int padding);
    // Add move constructor and move assignment operator
    ConvolutionLayer(ConvolutionLayer&& other) noexcept;
    ConvolutionLayer& operator=(ConvolutionLayer&& other) noexcept;

    // Delete copy constructor and copy assignment operator
    ConvolutionLayer(const ConvolutionLayer&) = delete;
    ConvolutionLayer& operator=(const ConvolutionLayer&) = delete;

private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    
#ifdef USE_CUDA
    std::unique_ptr<ConvGPU> gpuImplementation;
#endif
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [outChannels][inChannels][kernelSize][kernelSize]
    std::vector<float> biases; // [outChannels]
    std::vector<std::vector<std::vector<std::vector<float>>>> gradWeights;
    std::vector<float> gradBiases;
    std::vector<std::vector<float>> inputDataBatch;
    std::vector<std::vector<float>> outputDataBatch;
    std::vector<std::vector<AdamOptimizer>> weightOptimizers;
    AdamOptimizer biasOptimizer;
    std::vector<std::vector<float>> conv2DCPU(const std::vector<std::vector<float>>& inputBatch, int imageHeight, int imageWidth);
    void initializeWeights();
};

#endif // CONV_H