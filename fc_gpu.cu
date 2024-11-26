// fc_gpu.cu
#include "fc_gpu.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward propagation kernel
__global__ void fcForwardKernel(const float* input, const float* weights, 
                               const float* bias, float* output,
                               int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.x;
    int output_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || output_idx >= output_size) return;
    
    float sum = bias[output_idx];
    for (int i = 0; i < input_size; i++) {
        sum += input[batch_idx * input_size + i] * 
               weights[output_idx * input_size + i];
    }
    
    output[batch_idx * output_size + output_idx] = sum;
}

// Backward propagation kernels
__global__ void fcBackwardInputKernel(const float* dZ, const float* weights,
                                     float* dA_prev, int batch_size,
                                     int input_size, int output_size) {
    int batch_idx = blockIdx.x;
    int input_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || input_idx >= input_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < output_size; i++) {
        sum += dZ[batch_idx * output_size + i] * 
               weights[i * input_size + input_idx];
    }
    
    dA_prev[batch_idx * input_size + input_idx] = sum;
}

__global__ void fcBackwardParamsKernel(const float* dZ, const float* input,
                                      float* dW, float* db,
                                      int batch_size, int input_size, int output_size) {
    int output_idx = blockIdx.x;
    int input_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (output_idx >= output_size || input_idx >= input_size) return;
    
    float weight_grad = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        weight_grad += dZ[b * output_size + output_idx] * 
                      input[b * input_size + input_idx];
    }
    dW[output_idx * input_size + input_idx] = weight_grad / batch_size;
    
    if (input_idx == 0) {
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_grad += dZ[b * output_size + output_idx];
        }
        db[output_idx] = bias_grad / batch_size;
    }
}


// FCLayerGPU implementation
FCLayerGPU::FCLayerGPU(int input_size, int output_size) 
    : input_size(input_size), output_size(output_size), 
      memory_allocated(false), allocated_batch_size(0), timestep(1) {
    
    // Initialize weights and biases on host
    h_weights.resize(output_size, std::vector<float>(input_size));
    h_bias.resize(output_size);
    
    // He initialization
    float stddev = sqrt(2.0f / input_size);
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            h_weights[i][j] = stddev * ((float)rand() / RAND_MAX * 2 - 1);
        }
        h_bias[i] = 0.01f;
    }
    
    // Allocate and initialize device memory for weights and biases
    size_t weights_size = output_size * input_size * sizeof(float);
    size_t bias_size = output_size * sizeof(float);
    
    // Flatten weights for GPU
    std::vector<float> flat_weights;
    flat_weights.reserve(output_size * input_size);
    for (const auto& row : h_weights) {
        flat_weights.insert(flat_weights.end(), row.begin(), row.end());
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights, weights_size));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    
    // Allocate Adam optimizer states
    CUDA_CHECK(cudaMalloc(&d_weights_m, weights_size));
    CUDA_CHECK(cudaMalloc(&d_weights_v, weights_size));
    CUDA_CHECK(cudaMalloc(&d_bias_m, bias_size));
    CUDA_CHECK(cudaMalloc(&d_bias_v, bias_size));
    
    // Initialize Adam states to zero
    CUDA_CHECK(cudaMemset(d_weights_m, 0, weights_size));
    CUDA_CHECK(cudaMemset(d_weights_v, 0, weights_size));
    CUDA_CHECK(cudaMemset(d_bias_m, 0, bias_size));
    CUDA_CHECK(cudaMemset(d_bias_v, 0, bias_size));
}

void FCLayerGPU::allocateMemory(int batch_size) {
    if (!memory_allocated || batch_size > allocated_batch_size) {
        // Free existing memory if any
        freeMemoryIfAllocated();
        
        // Allocate new memory
        size_t input_mem_size = batch_size * input_size * sizeof(float);
        size_t output_mem_size = batch_size * output_size * sizeof(float);
        size_t weight_grad_size = output_size * input_size * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_input, input_mem_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_mem_size));
        CUDA_CHECK(cudaMalloc(&d_dZ, output_mem_size));
        CUDA_CHECK(cudaMalloc(&d_dW, weight_grad_size));
        CUDA_CHECK(cudaMalloc(&d_db, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dA_prev, input_mem_size));
        
        memory_allocated = true;
        allocated_batch_size = batch_size;
    }
}

void FCLayerGPU::freeMemoryIfAllocated() {
    if (memory_allocated) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_dZ));
        CUDA_CHECK(cudaFree(d_dW));
        CUDA_CHECK(cudaFree(d_db));
        CUDA_CHECK(cudaFree(d_dA_prev));
        memory_allocated = false;
    }
}

FCLayerGPU::~FCLayerGPU() {
    freeMemoryIfAllocated();
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_weights_m));
    CUDA_CHECK(cudaFree(d_weights_v));
    CUDA_CHECK(cudaFree(d_bias_m));
    CUDA_CHECK(cudaFree(d_bias_v));
}

std::vector<std::vector<float>> FCLayerGPU::forward(
    const std::vector<std::vector<float>>& input) {
    
    int batch_size = input.size();
    allocateMemory(batch_size);
    
    // Flatten input and copy to device
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * input_size);
    for (const auto& row : input) {
        flat_input.insert(flat_input.end(), row.begin(), row.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(), 
                         batch_size * input_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Launch forward kernel
    dim3 blockDim(256);
    dim3 gridDim(batch_size, (output_size + blockDim.x - 1) / blockDim.x);
    
    fcForwardKernel<<<gridDim, blockDim>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, input_size, output_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> flat_output(batch_size * output_size);
    CUDA_CHECK(cudaMemcpy(flat_output.data(), d_output,
                         batch_size * output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Reshape output
    std::vector<std::vector<float>> output(batch_size, 
                                         std::vector<float>(output_size));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            output[i][j] = flat_output[i * output_size + j];
        }
    }
    
    return output;
}

std::vector<std::vector<float>> FCLayerGPU::backward(
    const std::vector<std::vector<float>>& dZ) {
    
    int batch_size = dZ.size();
    
    // Flatten and copy dZ to device
    std::vector<float> flat_dZ;
    flat_dZ.reserve(batch_size * output_size);
    for (const auto& row : dZ) {
        flat_dZ.insert(flat_dZ.end(), row.begin(), row.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_dZ, flat_dZ.data(),
                         batch_size * output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Launch backward kernels
    dim3 blockDim(256);
    
    // Compute dA_prev
    dim3 gridDim1(batch_size, (input_size + blockDim.x - 1) / blockDim.x);
    fcBackwardInputKernel<<<gridDim1, blockDim>>>(
        d_dZ, d_weights, d_dA_prev,
        batch_size, input_size, output_size
    );
    
    // Compute dW and db
    dim3 gridDim2(output_size, (input_size + blockDim.x - 1) / blockDim.x);
    fcBackwardParamsKernel<<<gridDim2, blockDim>>>(
        d_dZ, d_input, d_dW, d_db,
        batch_size, input_size, output_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Update weights and biases using Adam
    int weight_size = output_size * input_size;
    int threads_per_block = 256;
    
    // Update weights
    int num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    adamUpdateKernel<<<num_blocks, threads_per_block>>>(
        d_weights, d_dW,
        d_weights_m, d_weights_v,
        lr, beta1, beta2, epsilon,
        weight_size, timestep
    );
    
    // Update biases
    num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    adamUpdateKernel<<<num_blocks, threads_per_block>>>(
        d_bias, d_db,
        d_bias_m, d_bias_v,
        lr, beta1, beta2, epsilon,
        output_size, timestep
    );
    
    timestep++;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy dA_prev back to host and reshape
    std::vector<float> flat_dA_prev(batch_size * input_size);
    CUDA_CHECK(cudaMemcpy(flat_dA_prev.data(), d_dA_prev,
                         batch_size * input_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    std::vector<std::vector<float>> dA_prev(batch_size,
                                           std::vector<float>(input_size));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < input_size; j++) {
            dA_prev[i][j] = flat_dA_prev[i * input_size + j];
        }
    }
    
    return dA_prev;
}
std::vector<std::vector<float>> FCLayerGPU::getWeights() const {
    std::vector<float> flat_weights(output_size * input_size);
    std::vector<float> biases(output_size);
    copyWeightsToHost(flat_weights, biases);

    std::vector<std::vector<float>> weights(output_size, std::vector<float>(input_size));
    for(int i = 0; i < output_size; ++i) {
        for(int j = 0; j < input_size; ++j) {
            weights[i][j] = flat_weights[i * input_size + j];
        }
    }
    return weights;
}

std::vector<float> FCLayerGPU::getBiases() const {
    std::vector<float> biases(output_size);
    std::vector<float> flat_weights(output_size * input_size);
    copyWeightsToHost(flat_weights, biases);
    return biases;
}

void FCLayerGPU::copyWeightsToHost(std::vector<float>& weights, std::vector<float>& biases) const {
    CUDA_CHECK(cudaMemcpy(weights.data(), d_weights, 
                         weights.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(biases.data(), d_bias, 
                         biases.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost));
}
