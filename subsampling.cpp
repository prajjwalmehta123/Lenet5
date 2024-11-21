#include "subsampling.h"

using namespace std;
subsampling::subsampling() {

}

subsampling::subsampling(int kernel_size, int stride, int image_kernel_size) : kernel_size(kernel_size), stride(stride), image_kernel_size(image_kernel_size) {
}

std::vector<std::vector<float>> subsampling::average_pooling(const vector<vector<float>>& inputBatch) {

    inputDataBatch = inputBatch;
    size_t batch_size = inputBatch.size();
    size_t totalInputSize = inputBatch[0].size();
    int featureSize = totalInputSize / image_kernel_size;
    int featureHeight = static_cast<int>(std::sqrt(featureSize));
    int featureWidth = featureHeight;

    int pooled_ht = (featureHeight - kernel_size) / stride + 1;
    int pooled_wdth = (featureWidth - kernel_size) / stride + 1;
    int pooledFeatureSize = pooled_ht * pooled_wdth;
    output_image_size = pooled_ht;
    int totalOutputSize = image_kernel_size * pooledFeatureSize;

    std::vector<std::vector<float>> output(batch_size, std::vector<float>(totalOutputSize, 0.0f));

    #pragma omp parallel for
    for (int image_idx = 0; image_idx < batch_size; ++image_idx) {
        const std::vector<float>& image = inputBatch[image_idx];
        std::vector<float> pooled_image(totalOutputSize, 0.0f);

        // Loop over each feature map
        for (int feature = 0; feature < image_kernel_size; ++feature) {
            int featureStartIndex = feature * featureSize;
            std::vector<float> featureMap(image.begin() + featureStartIndex, image.begin() + featureStartIndex + featureSize);

            std::vector<float> pooledFeatureMap(pooledFeatureSize);

            // Perform average pooling on the feature map
            for (int i = 0; i < pooled_ht; ++i) {
                for (int j = 0; j < pooled_wdth; ++j) {
                    float sum = 0.0f;
                    for (int m = 0; m < kernel_size; ++m) {
                        for (int n = 0; n < kernel_size; ++n) {
                            int rowIndex = i * stride + m;
                            int colIndex = j * stride + n;
                            int index = rowIndex * featureWidth + colIndex;
                            sum += featureMap[index];
                        }
                    }
                    pooledFeatureMap[i * pooled_wdth + j] = sum / (kernel_size * kernel_size);
                }
            }
            int pooledStartIndex = feature * pooledFeatureSize;
            std::copy(pooledFeatureMap.begin(), pooledFeatureMap.end(), pooled_image.begin() + pooledStartIndex);
        }

        output[image_idx] = pooled_image;
    }
    return output;
}

std::vector<std::vector<float>> subsampling::backward(const std::vector<std::vector<float>>& gradOutputBatch) {
    size_t batchSize = gradOutputBatch.size();
    size_t totalInputSize = inputDataBatch[0].size();

    // Initialize gradInputBatch with zeros
    std::vector<std::vector<float>> gradInputBatch(batchSize, std::vector<float>(totalInputSize, 0.0f));

    // Perform backpropagation
    #pragma omp parallel for
    for (size_t image_idx = 0; image_idx < batchSize; ++image_idx) {
        const std::vector<float>& gradOutputFlat = gradOutputBatch[image_idx];
        std::vector<float>& gradInputFlat = gradInputBatch[image_idx];

        // Loop over each feature map
        for (int feature = 0; feature < image_kernel_size; ++feature) {
            int featureInputStartIdx = feature * inputHeight * inputWidth;
            int featureOutputStartIdx = feature * pooledHeight * pooledWidth;

            // Loop over pooled feature map dimensions
            for (int ph = 0; ph < pooledHeight; ++ph) {
                for (int pw = 0; pw < pooledWidth; ++pw) {
                    int outputIdx = featureOutputStartIdx + ph * pooledWidth + pw;
                    float gradOutputValue = gradOutputFlat[outputIdx];

                    // Distribute gradient equally to each input in the pooling window
                    float gradInputValue = gradOutputValue / (kernel_size * kernel_size);

                    // Loop over the pooling window
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = ph * stride + kh;
                            int w_in = pw * stride + kw;
                            int inputIdx = featureInputStartIdx + h_in * inputWidth + w_in;

                            // Accumulate gradients
                            gradInputFlat[inputIdx] += gradInputValue;
                        }
                    }
                }
            }
        }
    }
    return gradInputBatch;
}


