// cuda_utils.cuh
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>

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

// Adam optimizer kernel declaration
__global__ void adamUpdateKernel(float* weights, float* gradients,
                               float* m, float* v,
                               float lr, float beta1, float beta2, float epsilon,
                               int size, int timestep);

#endif // CUDA_UTILS_CUH