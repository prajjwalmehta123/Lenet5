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
    // Layer objects
    ConvolutionLayer c1_layer, c3_layer;
    subsampling s2_layer, s4_layer;
    FCLayer f5_layer, f6_layer;
    Activation a1, a2, a3, a4, a5, a6;
    OutputLayer o1;

    // Cache intermediate outputs for debugging/visualization if needed
    struct LayerCache {
        std::vector<std::vector<float>> conv1_out, conv3_out;
        std::vector<std::vector<float>> pool2_out, pool4_out;
        std::vector<std::vector<float>> fc5_out, fc6_out;
    } cache;

    // Network configuration
    static constexpr int IMAGE_HEIGHT = 32;
    static constexpr int IMAGE_WIDTH = 32;
    static constexpr int CONV1_CHANNELS = 6;
    static constexpr int CONV3_CHANNELS = 16;
    static constexpr int FC5_NEURONS = 120;
    static constexpr int FC6_NEURONS = 84;
    static constexpr int OUTPUT_NEURONS = 10;

    // Store output and labels
    std::vector<std::vector<float>> logits;
    std::vector<int> predicted_labels;

    float computeLoss(const std::vector<int>& batch_labels) const;
    void computeAccuracy(const std::vector<int>& batch_labels, int& correct_count) const;

public:
    LeNet5();
    int Forward_Propagation(const std::vector<std::vector<float>>& batch_images,
                          const std::vector<int>& batch_labels);
    void Back_Propagation(const std::vector<int>& batch_labels);
};
#endif // LENET5_H