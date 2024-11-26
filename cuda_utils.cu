// cuda_utils.cu
#include <cmath>
#include "cuda_utils.cuh"

__global__ void adamUpdateKernel(float* weights, float* gradients,
                               float* m, float* v,
                               float lr, float beta1, float beta2, float epsilon,
                               int size, int timestep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float gradient = gradients[idx];
    float m_t = beta1 * m[idx] + (1 - beta1) * gradient;
    float v_t = beta2 * v[idx] + (1 - beta2) * gradient * gradient;
    
    m[idx] = m_t;
    v[idx] = v_t;
    
    float m_hat = m_t / (1.0f - powf(beta1, timestep));
    float v_hat = v_t / (1.0f - powf(beta2, timestep));
    
    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}