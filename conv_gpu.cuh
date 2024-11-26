// conv_gpu.cuh
#ifndef CONV_GPU_CUH
#define CONV_GPU_CUH

#include <vector>
#include <cuda_runtime.h>

class ConvGPU {
public:
    ConvGPU(int inputChannels, int outputChannels, int kernelSize, int stride, int padding);
    ~ConvGPU();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputBatch, 
                                          int imageHeight, int imageWidth);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutputBatch);
    std::vector<std::vector<std::vector<std::vector<float>>>> getWeights() const;
    void copyWeightsToHost(std::vector<float>& flat_weights, std::vector<float>& biases) const;
    std::vector<float> getBiases() const;
    void updateWeights();

private:
    // GPU memory pointers
    float *d_weights;
    float *d_biases;
    float *d_gradWeights;
    float *d_gradBiases;
    float *d_input;
    float *d_output;
    float *d_gradInput;
    float *d_gradOutput;
    float *d_weights_m;  // First moment vectors for weights
    float *d_weights_v;  // Second moment vectors for weights
    float *d_biases_m;   // First moment vectors for biases
    float *d_biases_v;   // Second moment vectors for biases
    int weights_timestep;
    int biases_timestep;
    
    // Adam hyperparameters
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    // Layer parameters
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    
    // Host memory for weights and biases
    std::vector<float> h_weights;
    std::vector<float> h_biases;
    
    void allocateMemory(int maxBatchSize, int maxImageSize);
    void freeMemory();
    
    // CUDA kernel wrapper functions
    void launchForwardKernel(int batchSize, int imageHeight, int imageWidth);
    void launchBackwardKernel(int batchSize, int imageHeight, int imageWidth);
    void launchUpdateWeightsKernel();

    size_t allocated_input_size;
    size_t allocated_output_size;
    bool memory_allocated;

    // Memory management functions
    void freeMemoryIfAllocated();
    bool needsReallocation(size_t required_input_size, size_t required_output_size);
};

#endif // CONV_GPU_CUH