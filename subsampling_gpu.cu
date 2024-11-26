// subsampling_gpu.cu
#include "subsampling_gpu.cuh"
#include <cmath>
#include <cstdio>   // Add this for fprintf and stderr
#include <cstdlib>  // Add this for exit

// Forward kernel for average pooling
__global__ void averagePoolForwardKernel(
    const float* input, float* output,
    int batchSize, int numFeatures,
    int inputHeight, int inputWidth,
    int pooledHeight, int pooledWidth,
    int kernelSize, int stride) {
    
    int batch = blockIdx.x;
    int feature = blockIdx.y;
    int row = threadIdx.x + blockDim.x * blockIdx.z;
    int col = threadIdx.y;

    if (batch >= batchSize || feature >= numFeatures ||
        row >= pooledHeight || col >= pooledWidth) return;

    int input_start_idx = (batch * numFeatures + feature) * (inputHeight * inputWidth);
    int output_idx = (batch * numFeatures + feature) * (pooledHeight * pooledWidth) +
                    row * pooledWidth + col;

    float sum = 0.0f;
    float scale = 1.0f / (kernelSize * kernelSize);

    #pragma unroll
    for (int kh = 0; kh < kernelSize; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernelSize; kw++) {
            int h_in = row * stride + kh;
            int w_in = col * stride + kw;
            int input_idx = input_start_idx + h_in * inputWidth + w_in;
            sum += input[input_idx];
        }
    }

    output[output_idx] = sum * scale;
}

// Backward kernel for average pooling
__global__ void averagePoolBackwardKernel(
    const float* gradOutput, float* gradInput,
    int batchSize, int numFeatures,
    int inputHeight, int inputWidth,
    int pooledHeight, int pooledWidth,
    int kernelSize, int stride) {
    
    int batch = blockIdx.x;
    int feature = blockIdx.y;
    int row = threadIdx.x + blockDim.x * blockIdx.z;
    int col = threadIdx.y;

    if (batch >= batchSize || feature >= numFeatures ||
        row >= inputHeight || col >= inputWidth) return;

    float gradValue = 0.0f;
    float scale = 1.0f / (kernelSize * kernelSize);

    // Find all pooling windows that include this input element
    int ph_start = (row < kernelSize) ? 0 : (row - kernelSize + stride) / stride;
    int pw_start = (col < kernelSize) ? 0 : (col - kernelSize + stride) / stride;
    int ph_end = min(row / stride + 1, pooledHeight);
    int pw_end = min(col / stride + 1, pooledWidth);

    int gradOutput_start_idx = (batch * numFeatures + feature) * (pooledHeight * pooledWidth);
    
    #pragma unroll
    for (int ph = ph_start; ph < ph_end; ph++) {
        #pragma unroll
        for (int pw = pw_start; pw < pw_end; pw++) {
            // Check if this input position is in the pooling window
            if (row >= ph * stride && row < ph * stride + kernelSize &&
                col >= pw * stride && col < pw * stride + kernelSize) {
                int gradOutput_idx = gradOutput_start_idx + ph * pooledWidth + pw;
                gradValue += gradOutput[gradOutput_idx] * scale;
            }
        }
    }

    int gradInput_idx = (batch * numFeatures + feature) * (inputHeight * inputWidth) +
                       row * inputWidth + col;
    gradInput[gradInput_idx] = gradValue;
}

SubsamplingGPU::SubsamplingGPU(int kernel_size, int stride, int image_size, int num_feature_maps)
    : kernel_size(kernel_size), stride(stride), image_size(image_size), 
      num_feature_maps(num_feature_maps), memory_allocated(false),
      allocated_batch_size(0), inputHeight(image_size), inputWidth(image_size) {
    
    pooledHeight = (image_size - kernel_size) / stride + 1;
    pooledWidth = (image_size - kernel_size) / stride + 1;
    output_image_size = pooledHeight;
}

SubsamplingGPU::~SubsamplingGPU() {
    freeMemoryIfAllocated();
}

void SubsamplingGPU::allocateMemory(int batch_size) {
    if (!memory_allocated || batch_size > allocated_batch_size) {
        freeMemoryIfAllocated();
        
        size_t input_size = batch_size * num_feature_maps * inputHeight * inputWidth;
        size_t output_size = batch_size * num_feature_maps * pooledHeight * pooledWidth;
        
        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradInput, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradOutput, output_size * sizeof(float)));
        
        memory_allocated = true;
        allocated_batch_size = batch_size;
    }
}

void SubsamplingGPU::freeMemoryIfAllocated() {
    if (memory_allocated) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_gradInput));
        CUDA_CHECK(cudaFree(d_gradOutput));
        memory_allocated = false;
    }
}

std::vector<std::vector<float>> SubsamplingGPU::forward(
    const std::vector<std::vector<float>>& inputBatch) {
    
    int batch_size = inputBatch.size();
    allocateMemory(batch_size);
    
    // Flatten and copy input to device
    std::vector<float> flat_input;
    size_t input_elements = batch_size * num_feature_maps * inputHeight * inputWidth;
    flat_input.reserve(input_elements);
    for (const auto& batch : inputBatch) {
        flat_input.insert(flat_input.end(), batch.begin(), batch.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(), 
                         input_elements * sizeof(float), 
                         cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(batch_size, 
                 num_feature_maps, 
                 (pooledHeight + blockDim.x - 1) / blockDim.x);
    
    averagePoolForwardKernel<<<gridDim, blockDim>>>(
        d_input, d_output,
        batch_size, num_feature_maps,
        inputHeight, inputWidth,
        pooledHeight, pooledWidth,
        kernel_size, stride
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    size_t output_elements = batch_size * num_feature_maps * pooledHeight * pooledWidth;
    std::vector<float> flat_output(output_elements);
    CUDA_CHECK(cudaMemcpy(flat_output.data(), d_output,
                         output_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Reshape output
    std::vector<std::vector<float>> outputBatch(batch_size);
    size_t elements_per_batch = num_feature_maps * pooledHeight * pooledWidth;
    for (int i = 0; i < batch_size; i++) {
        outputBatch[i].assign(
            flat_output.begin() + i * elements_per_batch,
            flat_output.begin() + (i + 1) * elements_per_batch
        );
    }
    
    return outputBatch;
}

std::vector<std::vector<float>> SubsamplingGPU::backward(
    const std::vector<std::vector<float>>& gradOutputBatch) {
    
    int batch_size = gradOutputBatch.size();
    
    // Copy gradient output to device
    std::vector<float> flat_grad_output;
    size_t output_elements = batch_size * num_feature_maps * pooledHeight * pooledWidth;
    flat_grad_output.reserve(output_elements);
    for (const auto& batch : gradOutputBatch) {
        flat_grad_output.insert(flat_grad_output.end(), batch.begin(), batch.end());
    }
    
    CUDA_CHECK(cudaMemcpy(d_gradOutput, flat_grad_output.data(),
                         output_elements * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(batch_size,
                 num_feature_maps,
                 (inputHeight + blockDim.x - 1) / blockDim.x);
    
    averagePoolBackwardKernel<<<gridDim, blockDim>>>(
        d_gradOutput, d_gradInput,
        batch_size, num_feature_maps,
        inputHeight, inputWidth,
        pooledHeight, pooledWidth,
        kernel_size, stride
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    size_t input_elements = batch_size * num_feature_maps * inputHeight * inputWidth;
    std::vector<float> flat_grad_input(input_elements);
    CUDA_CHECK(cudaMemcpy(flat_grad_input.data(), d_gradInput,
                         input_elements * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Reshape gradient input
    std::vector<std::vector<float>> gradInputBatch(batch_size);
    size_t elements_per_batch = num_feature_maps * inputHeight * inputWidth;
    for (int i = 0; i < batch_size; i++) {
        gradInputBatch[i].assign(
            flat_grad_input.begin() + i * elements_per_batch,
            flat_grad_input.begin() + (i + 1) * elements_per_batch
        );
    }
    
    return gradInputBatch;
}