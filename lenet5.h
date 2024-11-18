#ifndef LENET5_H
#define LENET5_H

#include <vector>
#include <string>
#include <map>
#include "FCLayer.h"
#include "conv.h"
#include "activation.h"
#include "subsampling.h"

class LeNet5 {
private:
    FCLayer F6; // Fully connected layer
    ConvolutionLayer c1_layer,c3_layer;
    subsampling s2_layer,s4_layer;

    // Kernel shapes for various layers
    std::map<std::string, std::vector<int>> kernel_shape = {
        {"C1", {5, 5, 1, 6}},
        {"C3", {5, 5, 6, 16}},
        {"C5", {5, 5, 16, 120}},
        {"F6", {120, 84}},
        {"OUTPUT", {84, 10}}
    };
    int imageHeight = 32;
    int imageWidth = 32;

    // Hyperparameters
    std::map<std::string, int> hparameters_convlayer = {{"stride", 1}, {"pad", 0}};
    std::map<std::string, int> hparameters_pooling = {{"stride", 2}, {"f", 2}};

    // Label for caching during forward propagation
    // std::vector<int> batch_labels;

public:
    // Constructor
    LeNet5();

    // Forward propagation
    void Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels);

    // Back propagation
    void Back_Propagation(float momentum, float weight_decay);

    // Stochastic Diagonal Levenberg-Marquardt (SDLM)
    void SDLM(float mu, float lr_global);

    // Initialize weights
    std::pair<std::vector<std::vector<float>>, std::vector<float>> initialize_weights(std::vector<int> kernel_shape);
    std::vector<std::vector<float>> flattenTensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& a3_FP);
    std::vector<int> Output_Layer(std::vector<std::vector<float>> X);
};

#endif // LENET5_H