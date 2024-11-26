// output_gpu.cuh
#ifndef OUTPUT_GPU_CUH
#define OUTPUT_GPU_CUH

#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class OutputLayerGPU {
public:
    OutputLayerGPU(int outputSize, int inputSize);
    ~OutputLayerGPU();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLoss);

private:
    // Layer dimensions
    int outputSize;
    int inputSize;
    
    // Device pointers
    float *d_weights;
    float *d_biases;
    float *d_input;
    float *d_output;
    float *d_temp;  // For intermediate computations
    float *d_dLoss;
    float *d_dInput;
    float *d_dWeights;
    float *d_dBiases;
    
    // Adam optimizer states
    float *d_weights_m;
    float *d_weights_v;
    float *d_biases_m;
    float *d_biases_v;
    int timestep;
    
    // Memory management
    bool memory_allocated;
    size_t allocated_batch_size;
    
    // Adam hyperparameters
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    
    // Helper functions
    void allocateMemory(int batch_size);
    void freeMemoryIfAllocated();
};

#endif // OUTPUT_GPU_CUH