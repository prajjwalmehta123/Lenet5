// activation_gpu.cuh
#ifndef ACTIVATION_GPU_CUH
#define ACTIVATION_GPU_CUH

#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class ActivationGPU {
public:
    ActivationGPU();
    ~ActivationGPU();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dZ);

private:
    // Device memory pointers
    float *d_input;       // Cached input for backward pass
    float *d_output;      // Forward pass output
    float *d_dZ;         // Gradient from next layer
    float *d_dA;         // Gradient to previous layer

    // Memory management
    bool memory_allocated;
    size_t allocated_batch_size;
    size_t allocated_feature_size;

    void allocateMemory(int batch_size, int feature_size);
    void freeMemoryIfAllocated();
};

#endif // ACTIVATION_GPU_CUH