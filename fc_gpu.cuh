// fc_gpu.cuh
#ifndef FC_GPU_CUH
#define FC_GPU_CUH

#include <vector>
#include <cuda_runtime.h>

class FCLayerGPU {
public:
    FCLayerGPU(int input_size, int output_size);
    ~FCLayerGPU();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dZ);
    std::vector<std::vector<float>> getWeights() const;
    std::vector<float> getBiases() const;

private:
    // Device pointers
    float *d_weights;
    float *d_bias;
    float *d_input;
    float *d_output;
    float *d_dZ;
    float *d_dW;
    float *d_db;
    float *d_dA_prev;
    
    // Adam optimizer states
    float *d_weights_m;
    float *d_weights_v;
    float *d_bias_m;
    float *d_bias_v;
    int timestep;
    
    // Host copies of weights and biases
    std::vector<std::vector<float>> h_weights;
    std::vector<float> h_bias;
    void copyWeightsToHost(std::vector<float>& weights, std::vector<float>& biases) const;
    
    // Dimensions
    int input_size;
    int output_size;
    
    // Memory tracking
    bool memory_allocated;
    size_t allocated_batch_size;
    
    // Helper functions
    void allocateMemory(int batch_size);
    void freeMemoryIfAllocated();
    
    // Adam hyperparameters
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
};

#endif // FC_GPU_CUH