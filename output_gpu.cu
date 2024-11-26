// output_gpu.cu
#include "output_gpu.cuh"
#include <cmath>
#include <cstdio>   // Add this for fprintf and stderr
#include <cstdlib>  // Add this for exit
#include "cuda_utils.cuh"

// Matrix multiplication kernel for forward pass
__global__ void matrixMulKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size,
    int input_size,
    int output_size) {
    
    int sample = blockIdx.x;
    int out_idx = threadIdx.x + blockDim.x * blockIdx.y;
    
    if (sample >= batch_size || out_idx >= output_size) return;
    
    float sum = biases[out_idx];
    for (int i = 0; i < input_size; i++) {
        sum += input[sample * input_size + i] * 
               weights[out_idx * input_size + i];
    }
    
    output[sample * output_size + out_idx] = sum;
}

// Softmax kernel
__global__ void softmaxKernel(
    float* input,
    float* output,
    int batch_size,
    int output_size) {
    
    int sample = blockIdx.x;
    if (sample >= batch_size) return;
    
    // Find max value for numerical stability
    float max_val = input[sample * output_size];
    for (int i = 1; i < output_size; i++) {
        float val = input[sample * output_size + i];
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float exp_val = expf(input[sample * output_size + i] - max_val);
        output[sample * output_size + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < output_size; i++) {
        output[sample * output_size + i] *= inv_sum;
    }
}

// Backward kernels
__global__ void gradientKernel(
    const float* input,
    const float* weights,
    const float* dLoss,
    float* dWeights,
    float* dBiases,
    float* dInput,
    int batch_size,
    int input_size,
    int output_size) {
    
    int sample = blockIdx.x;
    int out_idx = threadIdx.x + blockDim.x * blockIdx.y;
    
    if (sample >= batch_size || out_idx >= output_size) return;
    
    float dL = dLoss[sample * output_size + out_idx];
    
    // Compute gradients for weights and biases
    for (int i = 0; i < input_size; i++) {
        atomicAdd(&dWeights[out_idx * input_size + i],
                 dL * input[sample * input_size + i] / batch_size);
    }
    atomicAdd(&dBiases[out_idx], dL / batch_size);
    
    // Compute gradients for input
    for (int i = 0; i < input_size; i++) {
        atomicAdd(&dInput[sample * input_size + i],
                 dL * weights[out_idx * input_size + i]);
    }
}

OutputLayerGPU::OutputLayerGPU(int outputSize, int inputSize)
    : outputSize(outputSize), inputSize(inputSize),
      memory_allocated(false), allocated_batch_size(0), timestep(1) {
    
    // Allocate and initialize weights and biases
    size_t weights_size = outputSize * inputSize * sizeof(float);
    size_t bias_size = outputSize * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_weights, weights_size));
    CUDA_CHECK(cudaMalloc(&d_biases, bias_size));
    
    // Xavier initialization
    std::vector<float> h_weights(outputSize * inputSize);
    std::vector<float> h_biases(outputSize, 0.0f);
    
    float limit = sqrt(6.0f / (inputSize + outputSize));
    for (int i = 0; i < outputSize * inputSize; i++) {
        h_weights[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
    }
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, h_biases.data(), bias_size, cudaMemcpyHostToDevice));
    
    // Allocate Adam optimizer states
    CUDA_CHECK(cudaMalloc(&d_weights_m, weights_size));
    CUDA_CHECK(cudaMalloc(&d_weights_v, weights_size));
    CUDA_CHECK(cudaMalloc(&d_biases_m, bias_size));
    CUDA_CHECK(cudaMalloc(&d_biases_v, bias_size));
    
    // Initialize Adam states to zero
    CUDA_CHECK(cudaMemset(d_weights_m, 0, weights_size));
    CUDA_CHECK(cudaMemset(d_weights_v, 0, weights_size));
    CUDA_CHECK(cudaMemset(d_biases_m, 0, bias_size));
    CUDA_CHECK(cudaMemset(d_biases_v, 0, bias_size));
}

OutputLayerGPU::~OutputLayerGPU() {
    freeMemoryIfAllocated();
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_biases));
    CUDA_CHECK(cudaFree(d_weights_m));
    CUDA_CHECK(cudaFree(d_weights_v));
    CUDA_CHECK(cudaFree(d_biases_m));
    CUDA_CHECK(cudaFree(d_biases_v));
}

void OutputLayerGPU::allocateMemory(int batch_size) {
    if (!memory_allocated || batch_size > allocated_batch_size) {
        freeMemoryIfAllocated();
        
        size_t input_size = batch_size * inputSize * sizeof(float);
        size_t output_size = batch_size * outputSize * sizeof(float);
        size_t weights_grad_size = outputSize * inputSize * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMalloc(&d_temp, output_size));
        CUDA_CHECK(cudaMalloc(&d_dLoss, output_size));
        CUDA_CHECK(cudaMalloc(&d_dInput, input_size));
        CUDA_CHECK(cudaMalloc(&d_dWeights, weights_grad_size));
        CUDA_CHECK(cudaMalloc(&d_dBiases, outputSize * sizeof(float)));
        
        memory_allocated = true;
        allocated_batch_size = batch_size;
    }
}

void OutputLayerGPU::freeMemoryIfAllocated() {
    if (memory_allocated) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaFree(d_dLoss));
        CUDA_CHECK(cudaFree(d_dInput));
        CUDA_CHECK(cudaFree(d_dWeights));
        CUDA_CHECK(cudaFree(d_dBiases));
        memory_allocated = false;
    }
}

std::vector<std::vector<float>> OutputLayerGPU::forward(
    const std::vector<std::vector<float>>& input) {
    
    int batch_size = input.size();
    allocateMemory(batch_size);
    
    // Flatten and copy input
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * inputSize);
    for (const auto& sample : input) {
        flat_input.insert(flat_input.end(), sample.begin(), sample.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(),
                         batch_size * inputSize * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Matrix multiplication
    dim3 blockDim(256);
    dim3 gridDim(batch_size, (outputSize + blockDim.x - 1) / blockDim.x);
    
    matrixMulKernel<<<gridDim, blockDim>>>(
        d_input, d_weights, d_biases, d_temp,
        batch_size, inputSize, outputSize
    );
    
    // Softmax
    softmaxKernel<<<batch_size, 1>>>(
        d_temp, d_output,
        batch_size, outputSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<float> flat_output(batch_size * outputSize);
    CUDA_CHECK(cudaMemcpy(flat_output.data(), d_output,
                         batch_size * outputSize * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Reshape output
    std::vector<std::vector<float>> output(batch_size);
    for (int i = 0; i < batch_size; i++) {
        output[i].assign(
            flat_output.begin() + i * outputSize,
            flat_output.begin() + (i + 1) * outputSize
        );
    }
    
    return output;
}

std::vector<std::vector<float>> OutputLayerGPU::backward(
    const std::vector<std::vector<float>>& dLoss) {
    
    int batch_size = dLoss.size();
    
    // Copy gradient to device
    std::vector<float> flat_dLoss;
    flat_dLoss.reserve(batch_size * outputSize);
    for (const auto& sample : dLoss) {
        flat_dLoss.insert(flat_dLoss.end(), sample.begin(), sample.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_dLoss, flat_dLoss.data(),
                         batch_size * outputSize * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Reset gradients
    CUDA_CHECK(cudaMemset(d_dWeights, 0, outputSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dBiases, 0, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dInput, 0, batch_size * inputSize * sizeof(float)));
    
    // Compute gradients
    dim3 blockDim(256);
    dim3 gridDim(batch_size, (outputSize + blockDim.x - 1) / blockDim.x);
    
    gradientKernel<<<gridDim, blockDim>>>(
        d_input, d_weights, d_dLoss,
        d_dWeights, d_dBiases, d_dInput,
        batch_size, inputSize, outputSize
    );
    
    // Update weights and biases using Adam
    int weight_size = outputSize * inputSize;
    int threads_per_block = 256;
    int num_blocks;
    
    // Update weights
    num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    adamUpdateKernel<<<num_blocks, threads_per_block>>>(
        d_weights, d_dWeights,
        d_weights_m, d_weights_v,
        lr, beta1, beta2, epsilon,
        weight_size, timestep
    );
    
    // Update biases
    num_blocks = (outputSize + threads_per_block - 1) / threads_per_block;
    adamUpdateKernel<<<num_blocks, threads_per_block>>>(
        d_biases, d_dBiases,
        d_biases_m, d_biases_v,
        lr, beta1, beta2, epsilon,
        outputSize, timestep
    );
    
    timestep++;
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy gradients back to host
    std::vector<float> flat_dInput(batch_size * inputSize);
    CUDA_CHECK(cudaMemcpy(flat_dInput.data(), d_dInput,
                         batch_size * inputSize * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Reshape gradients
    std::vector<std::vector<float>> dInput(batch_size);
    for (int i = 0; i < batch_size; i++) {
        dInput[i].assign(
            flat_dInput.begin() + i * inputSize,
            flat_dInput.begin() + (i + 1) * inputSize
        );
    }
    
    return dInput;
}