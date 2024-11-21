#ifndef LENET5_H
#define LENET5_H

#include <vector>
#include <string>
#include <map>
#include "FCLayer.h"
#include "conv.h"
#include "activation.h"
#include "subsampling.h"
#include "out.h"


class LeNet5 {
private:
    FCLayer f5_layer,f6_layer; // Fully connected layer
    ConvolutionLayer c1_layer,c3_layer;
    subsampling s2_layer,s4_layer;
    Activation a1, a2, a3, a4;
    OutputLayer o1;

    // Kernel shapes for various layers
    std::map<std::string, std::vector<int>> kernel_shape = {
        {"C1", {5, 5, 1, 6}},
        {"C3", {5, 5, 6, 16}},
        {"F5", {120, 400}},
        {"F6", {84, 120}},
        {"OUTPUT", {10, 84}}
    };
    int imageHeight = 32;
    int imageWidth = 32;

    // Hyperparameters
    std::map<std::string, int> hparameters_convlayer = {{"stride", 1}, {"pad", 0}};
    std::map<std::string, int> hparameters_pooling = {{"stride", 2}, {"f", 2}};

    // Label for caching during forward propagation
    std::vector<int> labels;
    std::vector<std::vector<float>> logits;

public:
    // Constructor
    LeNet5();

    // Forward propagation
    void Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels);

    // Back propagation
    void Back_Propagation(std::vector<int>batch_labels);

    // Stochastic Diagonal Levenberg-Marquardt (SDLM)
    void SDLM(float mu, float lr_global);

    // Initialize weights
    std::pair<std::vector<std::vector<float>>, std::vector<float>> initialize_weights(std::vector<int> kernel_shape);
    std::vector<int> Output_Layer(std::vector<std::vector<float>> X, int outsize);
    void printShape(const std::vector<std::vector<float>>& tensor, const std::string& name);
};

#endif // LENET5_H