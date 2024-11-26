// activation_gpu.cu
#include "activation_gpu.cuh"
#include <cmath>
#include <cstdio>   // Add this for fprintf and stderr
#include <cstdlib>  // Add this for exit
// ReLU forward kernel
__global__ void reluForwardKernel(
    const float* input,
    float* output,
    int batch_size,
    int feature_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * feature_size) return;

    output[idx] = fmaxf(input[idx], 0.0f);
}

// ReLU backward kernel
__global__ void reluBackwardKernel(
    const float* dZ,
    const float* input,
    float* dA,
    int batch_size,
    int feature_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * feature_size) return;

    dA[idx] = dZ[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
}

ActivationGPU::ActivationGPU()
    : memory_allocated(false), allocated_batch_size(0), allocated_feature_size(0),
      d_input(nullptr), d_output(nullptr), d_dZ(nullptr), d_dA(nullptr) {
}

ActivationGPU::~ActivationGPU() {
    freeMemoryIfAllocated();
}

void ActivationGPU::allocateMemory(int batch_size, int feature_size) {
    if (!memory_allocated || 
        batch_size > allocated_batch_size || 
        feature_size > allocated_feature_size) {
        
        freeMemoryIfAllocated();
        
        size_t total_size = batch_size * feature_size;
        size_t bytes = total_size * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMalloc(&d_dZ, bytes));
        CUDA_CHECK(cudaMalloc(&d_dA, bytes));
        
        memory_allocated = true;
        allocated_batch_size = batch_size;
        allocated_feature_size = feature_size;
    }
}

void ActivationGPU::freeMemoryIfAllocated() {
    if (memory_allocated) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_dZ));
        CUDA_CHECK(cudaFree(d_dA));
        
        d_input = nullptr;
        d_output = nullptr;
        d_dZ = nullptr;
        d_dA = nullptr;
        
        memory_allocated = false;
    }
}

std::vector<std::vector<float>> ActivationGPU::forward(
    const std::vector<std::vector<float>>& input) {
    
    int batch_size = input.size();
    int feature_size = input[0].size();
    
    allocateMemory(batch_size, feature_size);
    
    // Flatten and copy input to device
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * feature_size);
    for (const auto& batch : input) {
        flat_input.insert(flat_input.end(), batch.begin(), batch.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(),
                         batch_size * feature_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size * feature_size + threads_per_block - 1) / threads_per_block;
    
    reluForwardKernel<<<num_blocks, threads_per_block>>>(
        d_input, d_output, batch_size, feature_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> flat_output(batch_size * feature_size);
    CUDA_CHECK(cudaMemcpy(flat_output.data(), d_output,
                         batch_size * feature_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Reshape output
    std::vector<std::vector<float>> output(batch_size);
    for (int i = 0; i < batch_size; i++) {
        output[i].assign(
            flat_output.begin() + i * feature_size,
            flat_output.begin() + (i + 1) * feature_size
        );
    }
    
    return output;
}

std::vector<std::vector<float>> ActivationGPU::backward(
    const std::vector<std::vector<float>>& dZ) {
    
    int batch_size = dZ.size();
    int feature_size = dZ[0].size();
    
    // Flatten and copy dZ to device
    std::vector<float> flat_dZ;
    flat_dZ.reserve(batch_size * feature_size);
    for (const auto& batch : dZ) {
        flat_dZ.insert(flat_dZ.end(), batch.begin(), batch.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_dZ, flat_dZ.data(),
                         batch_size * feature_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size * feature_size + threads_per_block - 1) / threads_per_block;
    
    reluBackwardKernel<<<num_blocks, threads_per_block>>>(
        d_dZ, d_input, d_dA, batch_size, feature_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> flat_dA(batch_size * feature_size);
    CUDA_CHECK(cudaMemcpy(flat_dA.data(), d_dA,
                         batch_size * feature_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Reshape gradient
    std::vector<std::vector<float>> dA(batch_size);
    for (int i = 0; i < batch_size; i++) {
        dA[i].assign(
            flat_dA.begin() + i * feature_size,
            flat_dA.begin() + (i + 1) * feature_size
        );
    }
    
    return dA;
}