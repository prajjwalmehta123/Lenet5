#ifndef SUBSAMPLING_H
#define SUBSAMPLING_H

#include <vector>
#include <omp.h>

class subsampling {
public:
    int output_image_size;
    subsampling();
    subsampling(int kernel_size, int stride, int image_kernel_size);

    // Forward pass
    std::vector<std::vector<float>> average_pooling(const std::vector<std::vector<float>>& inputBatch);

    // Backward pass
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradOutputBatch);

private:
    int kernel_size;
    int stride;
    int image_kernel_size;

    // Variables to store forward pass data
    std::vector<std::vector<float>> inputDataBatch; // Stores input data for backward pass
    int inputHeight;
    int inputWidth;
    int pooledHeight;
    int pooledWidth;
};

#endif // SUBSAMPLING_H
