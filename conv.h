#ifndef CONV_H
#define CONV_H
#include <vector>
#include <memory>
#include <iostream>

//Forward declaration for future GPU class
class GPUComputations;

class ConvolutionLayer {
public:
    ConvolutionLayer(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int padding = 0);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutput);
    void updateWeights(float learningRate);
    void setGPUComputations(std::shared_ptr<GPUComputations> gpuComputations);

private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [outChannels][inChannels][kernelHeight][kernelWidth]
    std::vector<float> biases; // [outChannels]
    std::vector<std::vector<float>> inputData;
    std::vector<std::vector<float>> outputData;
    // Pointer to GPU computations handler
    std::shared_ptr<GPUComputations> gpuComputations;
    void initializeWeights();
    std::vector<std::vector<float>> conv2DCPU(const std::vector<std::vector<float>>& input);
    int calculateOutputSize(int inputSize, int kernelSize, int stride, int padding);
};

#endif // CONV_H
