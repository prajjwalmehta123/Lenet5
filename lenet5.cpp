#include "lenet5.h"
#include <iostream>
#include <random>

// Constructor
LeNet5::LeNet5(){
    for (const auto& pair : kernel_shape) {
        std::cout << pair.first << ": ";
        for (const auto& val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    c1_layer = ConvolutionLayer(kernel_shape["C1"][2], kernel_shape["C1"][3], kernel_shape["C1"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    //ConvolutionLayer convLayer(kernel_shape["C1"][2], kernel_shape["C1"][3], kernel_shape["C1"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    s2_layer = subsampling(hparameters_pooling["f"],hparameters_pooling["stride"],kernel_shape["C1"][3]);
    //subsampling s2_layer(hparameters_pooling["f"],hparameters_pooling["stride"]);
    F6 = FCLayer({kernel_shape["F6"][0], kernel_shape["F6"][1]});

}

std::vector<std::vector<float>> LeNet5::flattenTensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& a3_FP) {
    size_t batchSize = a3_FP.size();
    size_t channels = a3_FP[0][0][0].size();
    std::vector<std::vector<float>> flattened(batchSize, std::vector<float>(channels));

    for (size_t i = 0; i < batchSize; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            flattened[i][c] = a3_FP[i][0][0][c];
        }
    }
    return flattened;
}

// Forward Propagation
void LeNet5::Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels) {
    std::vector<std::vector<float>> c1_out = c1_layer.forward(batch_images, imageHeight, imageWidth);
    std::vector<std::vector<float>> s2_out = s2_layer.average_pooling(c1_out);
    // std::vector<std::vector<float>> flattened = flattenTensor(a3_FP);
    // F6.forward_prop(flattened);
    //std::vector<std::vector<float>> f6_FP = F6.forward_prop(flattened);
    //Output_Layer(f6_FP)
}

// Back Propagation
void LeNet5::Back_Propagation(float momentum, float weight_decay) {
    // Placeholder for backpropagation logic
}

// Stochastic Diagonal Levenberg-Marquardt (SDLM)
void LeNet5::SDLM(float mu, float lr_global) {
    // Placeholder for SDLM logic
}

std::vector<int> LeNet5::Output_Layer(std::vector<std::vector<float>> X){
    int inp = kernel_shape["OUTPUT"][0];
    int out = kernel_shape["OUTPUT"][1];
    std::vector<int> Y(out);
    for (int i = 0; i < inp; i++) {
        int max_idx = 0;
        float max_val = X[i][max_idx];
        for (int j = 1; j < out; j++) {
            float elem = X[i][j];
            if (elem > max_val) {
                max_idx = j;
                max_val = elem;
            }
        }
        // Assign the index of the maximum value to the output
        Y[i] = max_idx;
    }
    return Y;
}


// Initialize weights
std::pair<std::vector<std::vector<float>>, std::vector<float>> LeNet5::initialize_weights(std::vector<int> kernel_shape) {
    std::vector<std::vector<float>> weight;
    std::vector<float> bias;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0, 0.1); // Mean = 0.0, Std Dev = 0.1

    // Initialize weight matrix (rows = kernel_shape[0], cols = kernel_shape[1])
    int rows = kernel_shape[0];
    int cols = kernel_shape[1];
    weight.resize(rows, std::vector<float>(cols, 0.0f)); // Initialize with zeros
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weight[i][j] = d(gen); // Assign random values from Gaussian distribution
        }
    }

    // Initialize bias vector (size = kernel_shape[1])
    bias.resize(cols, 0.01f); // Small constant value for bias initialization

    return {weight, bias};
}