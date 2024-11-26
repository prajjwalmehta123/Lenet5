// conv_gpu.cu
#include "conv_gpu.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>   // Add this for fprintf and stderr
#include <cstdlib>  // Add this for exit


// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward pass kernel
__global__ void convForwardKernel(const float* input, const float* weights, 
                                 const float* biases, float* output,
                                 int batchSize, int inputChannels, int outputChannels,
                                 int inputHeight, int inputWidth,
                                 int outputHeight, int outputWidth,
                                 int kernelSize, int stride, int padding) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int oh = threadIdx.x + blockDim.x * blockIdx.z;
    int ow = threadIdx.y;

    if (b >= batchSize || oc >= outputChannels || 
        oh >= outputHeight || ow >= outputWidth) return;

    float sum = biases[oc];
    
    for (int ic = 0; ic < inputChannels; ic++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                    int inputIdx = ((b * inputChannels + ic) * inputHeight + ih) * inputWidth + iw;
                    int weightIdx = ((oc * inputChannels + ic) * kernelSize + kh) * kernelSize + kw;
                    sum += input[inputIdx] * weights[weightIdx];
                }
            }
        }
    }

    int outputIdx = ((b * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;
    output[outputIdx] = sum;
}

// Backward pass kernel for input gradients
__global__ void convBackwardInputKernel(const float* gradOutput, const float* weights,
                                      float* gradInput, int batchSize,
                                      int inputChannels, int outputChannels,
                                      int inputHeight, int inputWidth,
                                      int outputHeight, int outputWidth,
                                      int kernelSize, int stride, int padding) {
    int b = blockIdx.x;
    int ic = blockIdx.y;
    int ih = threadIdx.x + blockDim.x * blockIdx.z;
    int iw = threadIdx.y;

    if (b >= batchSize || ic >= inputChannels ||
        ih >= inputHeight || iw >= inputWidth) return;

    float sum = 0.0f;

    for (int oc = 0; oc < outputChannels; oc++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                int oh = (ih + padding - kh) / stride;
                int ow = (iw + padding - kw) / stride;
                
                if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth &&
                    (ih + padding - kh) % stride == 0 && (iw + padding - kw) % stride == 0) {
                    int gradOutputIdx = ((b * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    int weightIdx = ((oc * inputChannels + ic) * kernelSize + kh) * kernelSize + kw;
                    sum += gradOutput[gradOutputIdx] * weights[weightIdx];
                }
            }
        }
    }

    int gradInputIdx = ((b * inputChannels + ic) * inputHeight + ih) * inputWidth + iw;
    gradInput[gradInputIdx] = sum;
}

// Backward pass kernel for weight gradients
__global__ void convBackwardWeightKernel(const float* input, const float* gradOutput,
                                       float* gradWeights, float* gradBiases,
                                       int batchSize, int inputChannels, int outputChannels,
                                       int inputHeight, int inputWidth,
                                       int outputHeight, int outputWidth,
                                       int kernelSize, int stride, int padding) {
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int kh = threadIdx.x;
    int kw = threadIdx.y;

    if (oc >= outputChannels || ic >= inputChannels ||
        kh >= kernelSize || kw >= kernelSize) return;

    float weightGradSum = 0.0f;
    float biasGradSum = 0.0f;

    for (int b = 0; b < batchSize; b++) {
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                    int inputIdx = ((b * inputChannels + ic) * inputHeight + ih) * inputWidth + iw;
                    int gradOutputIdx = ((b * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    weightGradSum += input[inputIdx] * gradOutput[gradOutputIdx];
                    if (ic == 0 && kh == 0 && kw == 0) {
                        biasGradSum += gradOutput[gradOutputIdx];
                    }
                }
            }
        }
    }

    int weightIdx = ((oc * inputChannels + ic) * kernelSize + kh) * kernelSize + kw;
    gradWeights[weightIdx] = weightGradSum / batchSize;
    
    if (ic == 0 && kh == 0 && kw == 0) {
        gradBiases[oc] = biasGradSum / batchSize;
    }
}

// ConvGPU implementation
ConvGPU::ConvGPU(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
    : inputChannels(inputChannels), outputChannels(outputChannels),
      kernelSize(kernelSize), stride(stride), padding(padding),
      weights_timestep(1), biases_timestep(1),
      memory_allocated(false), allocated_input_size(0), allocated_output_size(0),
      d_input(nullptr), d_output(nullptr), d_gradInput(nullptr), d_gradOutput(nullptr) {

    //CUDA_CHECK(cudaSetDevice(1));
    
    // Initialize weights and biases
    int weightSize = outputChannels * inputChannels * kernelSize * kernelSize;
    h_weights.resize(weightSize);
    h_biases.resize(outputChannels);

    // He initialization
    float stddev = sqrt(2.0f / (inputChannels * kernelSize * kernelSize));
    for (int i = 0; i < weightSize; i++) {
        h_weights[i] = stddev * ((float)rand() / RAND_MAX * 2 - 1);
    }
    for (int i = 0; i < outputChannels; i++) {
        h_biases[i] = 0.0f;
    }

    // Allocate GPU memory for weights and gradients
    CUDA_CHECK(cudaMalloc(&d_weights, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases, outputChannels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradWeights, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradBiases, outputChannels * sizeof(float)));

    // Copy initial weights and biases to GPU
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, h_biases.data(), outputChannels * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate Adam optimizer memory
    CUDA_CHECK(cudaMalloc(&d_weights_m, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_v, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases_m, outputChannels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases_v, outputChannels * sizeof(float)));
    
    // Initialize moments to zero
    CUDA_CHECK(cudaMemset(d_weights_m, 0, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weights_v, 0, weightSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_biases_m, 0, outputChannels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_biases_v, 0, outputChannels * sizeof(float)));
}
void ConvGPU::freeMemoryIfAllocated() {
    if (memory_allocated) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_gradInput));
        CUDA_CHECK(cudaFree(d_gradOutput));
        d_input = nullptr;
        d_output = nullptr;
        d_gradInput = nullptr;
        d_gradOutput = nullptr;
        memory_allocated = false;
        allocated_input_size = 0;
        allocated_output_size = 0;
    }
}
bool ConvGPU::needsReallocation(size_t required_input_size, size_t required_output_size) {
    return !memory_allocated ||
           required_input_size > allocated_input_size ||
           required_output_size > allocated_output_size;
}


ConvGPU::~ConvGPU() {
    // Free memory
    freeMemoryIfAllocated();
    
    // Free weights and gradients
    if (d_weights) CUDA_CHECK(cudaFree(d_weights));
    if (d_biases) CUDA_CHECK(cudaFree(d_biases));
    if (d_gradWeights) CUDA_CHECK(cudaFree(d_gradWeights));
    if (d_gradBiases) CUDA_CHECK(cudaFree(d_gradBiases));
    if (d_weights_m) CUDA_CHECK(cudaFree(d_weights_m));
    if (d_weights_v) CUDA_CHECK(cudaFree(d_weights_v));
    if (d_biases_m) CUDA_CHECK(cudaFree(d_biases_m));
    if (d_biases_v) CUDA_CHECK(cudaFree(d_biases_v));
}
std::vector<std::vector<float>> ConvGPU::forward(const std::vector<std::vector<float>>& inputBatch,
                                                int imageHeight, int imageWidth) {
    int batchSize = inputBatch.size();
    int outputHeight = (imageHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (imageWidth + 2 * padding - kernelSize) / stride + 1;
    
    // Allocate memory if needed
    allocateMemory(batchSize, std::max(imageHeight, imageWidth));

    // Copy input data to GPU
    size_t inputSize = batchSize * inputChannels * imageHeight * imageWidth;
    std::vector<float> flatInput;
    flatInput.reserve(inputSize);
    for (const auto& batch : inputBatch) {
        flatInput.insert(flatInput.end(), batch.begin(), batch.end());
    }
    CUDA_CHECK(cudaMemcpy(d_input, flatInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Launch forward kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(batchSize, outputChannels, (outputHeight + blockDim.x - 1) / blockDim.x);
    
    convForwardKernel<<<gridDim, blockDim>>>(
        d_input, d_weights, d_biases, d_output,
        batchSize, inputChannels, outputChannels,
        imageHeight, imageWidth, outputHeight, outputWidth,
        kernelSize, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    size_t outputSize = batchSize * outputChannels * outputHeight * outputWidth;
    std::vector<float> flatOutput(outputSize);
    CUDA_CHECK(cudaMemcpy(flatOutput.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Reshape output
    std::vector<std::vector<float>> outputBatch(batchSize);
    size_t outputSizePerBatch = outputChannels * outputHeight * outputWidth;
    for (int b = 0; b < batchSize; b++) {
        outputBatch[b].assign(flatOutput.begin() + b * outputSizePerBatch,
                            flatOutput.begin() + (b + 1) * outputSizePerBatch);
    }

    return outputBatch;
}

void ConvGPU::allocateMemory(int maxBatchSize, int maxImageSize) {
    // Calculate required sizes
    size_t required_input_size = maxBatchSize * inputChannels * maxImageSize * maxImageSize;
    size_t required_output_size = maxBatchSize * outputChannels * 
                               ((maxImageSize + 2 * padding - kernelSize) / stride + 1) *
                               ((maxImageSize + 2 * padding - kernelSize) / stride + 1);

    // Check if we need to reallocate
    if (needsReallocation(required_input_size, required_output_size)) {
        // Free existing memory if any
        freeMemoryIfAllocated();

        // Allocate new memory
        CUDA_CHECK(cudaMalloc(&d_input, required_input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, required_output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradInput, required_input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradOutput, required_output_size * sizeof(float)));

        allocated_input_size = required_input_size;
        allocated_output_size = required_output_size;
        memory_allocated = true;
    }
}

std::vector<std::vector<float>> ConvGPU::backward(const std::vector<std::vector<float>>& gradOutputBatch) {
    int batchSize = gradOutputBatch.size();
    int outputSize = gradOutputBatch[0].size();
    int outputHeight = static_cast<int>(sqrt(outputSize / outputChannels));
    int outputWidth = outputHeight;
    int inputHeight = (outputHeight - 1) * stride - 2 * padding + kernelSize;
    int inputWidth = inputHeight;

    // Copy gradient output to device
    size_t gradOutputSize = batchSize * outputChannels * outputHeight * outputWidth;
    std::vector<float> flatGradOutput;
    flatGradOutput.reserve(gradOutputSize);
    for (const auto& batch : gradOutputBatch) {
        flatGradOutput.insert(flatGradOutput.end(), batch.begin(), batch.end());
    }
    CUDA_CHECK(cudaMemcpy(d_gradOutput, flatGradOutput.data(), 
                         gradOutputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Launch backward kernels
    
    // 1. Compute gradients for input
    dim3 blockDim(16, 16);
    dim3 gridDim(batchSize, inputChannels, (inputHeight + blockDim.x - 1) / blockDim.x);
    
    convBackwardInputKernel<<<gridDim, blockDim>>>(
        d_gradOutput, d_weights, d_gradInput,
        batchSize, inputChannels, outputChannels,
        inputHeight, inputWidth, outputHeight, outputWidth,
        kernelSize, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Compute gradients for weights and biases
    dim3 weightBlockDim(kernelSize, kernelSize);
    dim3 weightGridDim(outputChannels, inputChannels);
    
    convBackwardWeightKernel<<<weightGridDim, weightBlockDim>>>(
        d_input, d_gradOutput, d_gradWeights, d_gradBiases,
        batchSize, inputChannels, outputChannels,
        inputHeight, inputWidth, outputHeight, outputWidth,
        kernelSize, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy gradient input back to host
    size_t gradInputSize = batchSize * inputChannels * inputHeight * inputWidth;
    std::vector<float> flatGradInput(gradInputSize);
    CUDA_CHECK(cudaMemcpy(flatGradInput.data(), d_gradInput, 
                         gradInputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Reshape gradient input
    std::vector<std::vector<float>> gradInputBatch(batchSize);
    size_t gradInputSizePerBatch = inputChannels * inputHeight * inputWidth;
    for (int b = 0; b < batchSize; b++) {
        gradInputBatch[b].assign(flatGradInput.begin() + b * gradInputSizePerBatch,
                               flatGradInput.begin() + (b + 1) * gradInputSizePerBatch);
    }

    updateWeights();
    return gradInputBatch;
}


void ConvGPU::updateWeights() {
    int weightSize = outputChannels * inputChannels * kernelSize * kernelSize;
    int threadsPerBlock = 256;
    
    // Update weights
    int numBlocks = (weightSize + threadsPerBlock - 1) / threadsPerBlock;
    adamUpdateKernel<<<numBlocks, threadsPerBlock>>>(
        d_weights, d_gradWeights,
        d_weights_m, d_weights_v,
        lr, beta1, beta2, epsilon,
        weightSize, weights_timestep
    );
    weights_timestep++;
    
    // Update biases
    numBlocks = (outputChannels + threadsPerBlock - 1) / threadsPerBlock;
    adamUpdateKernel<<<numBlocks, threadsPerBlock>>>(
        d_biases, d_gradBiases,
        d_biases_m, d_biases_v,
        lr, beta1, beta2, epsilon,
        outputChannels, biases_timestep
    );
    biases_timestep++;
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}