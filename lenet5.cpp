#include "lenet5.h"
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>

// Constructor
LeNet5::LeNet5(){
    c1_layer = ConvolutionLayer(kernel_shape["C1"][2], kernel_shape["C1"][3], kernel_shape["C1"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    a1 = Activation();
    s2_layer = subsampling(2,2,28,6);
    a2 = Activation();
    c3_layer = ConvolutionLayer(kernel_shape["C3"][2], kernel_shape["C3"][3], kernel_shape["C3"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    a3 = Activation();
    s4_layer = subsampling(2,2,10,16);
    a4 = Activation();
    f5_layer = FCLayer({kernel_shape["F5"][0], kernel_shape["F5"][1]});
    a5 = Activation();
    f6_layer = FCLayer({kernel_shape["F6"][0], kernel_shape["F6"][1]});
    a6 = Activation();
    o1 = OutputLayer({kernel_shape["OUTPUT"][0], kernel_shape["OUTPUT"][1]});
}


// Forward Propagation
void LeNet5::Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int> batch_labels) {
    using namespace std::chrono;
    std::vector<std::vector<float>> out = c1_layer.forward(batch_images, imageHeight, imageWidth);
    out = a1.forwardProp(out);
    out = s2_layer.average_pooling(out);
    out = a2.forwardProp(out);
    out = c3_layer.forward(out, s2_layer.output_image_size, s2_layer.output_image_size);
    out = a3.forwardProp(out);
    out = s4_layer.average_pooling(out);
    out = a4.forwardProp(out);
    out = f5_layer.forward_prop(out);
    out = a5.forwardProp(out);
    out = f6_layer.forward_prop(out);
    out = a6.forwardProp(out);
    logits = o1.forwardProp(out);
    labels = Output_Layer(logits, batch_images.size());
}

float LeNet5::compute_loss(std::vector<int>batch_labels) {
    float total_loss = 0.0f;

    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_labels.size(); ++i) {
        float prob = logits[i][batch_labels[i]];
        total_loss += -std::log(prob);
    }
    total_loss /= batch_labels.size();
    return total_loss;
}

int LeNet5::compute_accuracy(std::vector<int>batch_labels) {
    int correct = 0;
    for(int i = 0; i< labels.size();i++){
        if (labels[i] == batch_labels[i]) {
            correct++;
        }
    }
    return correct;
}

// Back Propagation
void LeNet5::Back_Propagation(std::vector<int> batch_labels) {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    std::vector<std::vector<float>> dy_pred(batch_labels.size(), std::vector<float>(kernel_shape["OUTPUT"][0], 0));
    for (size_t i = 0; i < batch_labels.size(); ++i) {
        for (size_t j = 0; j < logits[i].size(); ++j) {
            dy_pred[i][j] = logits[i][j];
            if (j == batch_labels[i]) {
                dy_pred[i][j] -= 1.0f;
            }
        }
    }
    std::vector<std::vector<float>> back_out = o1.backProp(dy_pred);
    back_out = a6.backProp(back_out);
    back_out = f6_layer.back_prop(back_out);
    back_out = a5.backProp(back_out);
    back_out = f5_layer.back_prop(back_out);
    back_out = a4.backProp(back_out);
    back_out = s4_layer.backward(back_out);
    back_out = a3.backProp(back_out);
    back_out = c3_layer.backward(back_out);
    back_out = a2.backProp(back_out);
    back_out = s2_layer.backward(back_out);
    back_out = a1.backProp(back_out);
    back_out = c1_layer.backward(back_out);
}

std::vector<int> LeNet5::Output_Layer(std::vector<std::vector<float>> X, int outsize) {
    int inp = kernel_shape["OUTPUT"][0];
    std::vector<int> Y(outsize, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < outsize; i++) {
        int max_idx = 0;
        float max_val = X[i][0]; // Initialize with the first element of the row
        for (int j = 1; j < inp; j++) {
            if (X[i][j] > max_val) {
                max_idx = j;
                max_val = X[i][j];
            }
        }
        Y[i] = max_idx; // Assign the index of the maximum value
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
    bias.resize(cols, 0.0f); // Small constant value for bias initialization
    return {weight, bias};
}