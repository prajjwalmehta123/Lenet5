// subsampling_gpu.cuh
#ifndef SUBSAMPLING_GPU_CUH
#define SUBSAMPLING_GPU_CUH

#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class SubsamplingGPU {
public:
    SubsamplingGPU(int kernel_size, int stride, int image_size, int num_feature_maps);
    ~SubsamplingGPU();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputBatch);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutputBatch);

    int getOutputSize() const { return output_image_size; }

private:
    // Layer parameters
    int kernel_size;
    int stride;
    int num_feature_maps;
    int image_size;
    int output_image_size;

    // Device memory pointers
    float *d_input;
    float *d_output;
    float *d_gradInput;
    float *d_gradOutput;

    // Memory management
    bool memory_allocated;
    size_t allocated_batch_size;
    
    // Helper functions
    void allocateMemory(int batch_size);
    void freeMemoryIfAllocated();

    // Dimensions
    int inputHeight;
    int inputWidth;
    int pooledHeight;
    int pooledWidth;
};

#endif // SUBSAMPLING_GPU_CUH