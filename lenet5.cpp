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
    a1 = Activation();
    //ConvolutionLayer convLayer(kernel_shape["C1"][2], kernel_shape["C1"][3], kernel_shape["C1"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    s2_layer = subsampling(hparameters_pooling["f"],hparameters_pooling["stride"],kernel_shape["C1"][3]);
    //subsampling s2_layer(hparameters_pooling["f"],hparameters_pooling["stride"]);
    c3_layer = ConvolutionLayer(kernel_shape["C3"][2], kernel_shape["C3"][3], kernel_shape["C3"][0], hparameters_convlayer["stride"], hparameters_convlayer["padding"]);
    a2 = Activation();
    s4_layer = subsampling(hparameters_pooling["f"],hparameters_pooling["stride"],kernel_shape["C3"][3]);
    f5_layer = FCLayer({kernel_shape["F5"][0], kernel_shape["F5"][1]});
    a3 = Activation();
    f6_layer = FCLayer({kernel_shape["F6"][0], kernel_shape["F6"][1]});
    a4 = Activation();
    o1 = OutputLayer({kernel_shape["OUTPUT"][0], kernel_shape["OUTPUT"][1]});
}

// Forward Propagation
void LeNet5::Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels) {
    std::vector<std::vector<float>> c1_out = c1_layer.forward(batch_images, imageHeight, imageWidth);
    printShape(c1_out, "c1_out");
    std::vector<std::vector<float>> a1_out = a1.forwardProp(c1_out);
    printShape(a1_out, "a1_out");
    std::vector<std::vector<float>> s2_out = s2_layer.average_pooling(a1_out);
    printShape(s2_out, "s2_out");
    std::vector<std::vector<float>> c3_out = c3_layer.forward(s2_out, s2_layer.output_image_size, s2_layer.output_image_size);
    printShape(c3_out, "c3_out");
    std::vector<std::vector<float>> a2_out = a2.forwardProp(c3_out);
    printShape(a2_out, "a2_out");
    std::vector<std::vector<float>> s4_out = s4_layer.average_pooling(a2_out);
    printShape(s4_out, "s4_out");
    std::vector<std::vector<float>> f5_out = f5_layer.forward_prop(s4_out);
    printShape(f5_out, "f5_out");
    std::vector<std::vector<float>> a3_out = a3.forwardProp(f5_out);
    printShape(a3_out, "a3_out");
    std::vector<std::vector<float>> f6_out = f6_layer.forward_prop(a3_out);
    printShape(f6_out, "f6_out");
    std::vector<std::vector<float>> a4_out = a4.forwardProp(f6_out);
    printShape(a4_out, "a4_out");
    logits = o1.forwardProp(a4_out);
    printShape(logits, "logits");
    labels = Output_Layer(logits,batch_images.size());
    std::cout<<"Label size: "<<labels.size();
    int correct = 0;
    for(int i = 0; i< labels.size();i++){
        if (labels[i] == batch_labels[i]) {
            correct++;
        }
    }
    std::cout<<"Batch Processed, Number of correct: "<<correct<<std::endl;
}

// Back Propagation
void LeNet5::Back_Propagation(std::vector<int>batch_labels) {
    std::vector<std::vector<float>> dy_pred(batch_labels.size(), std::vector<float>(kernel_shape["OUTPUT"][1], 0)); // Properly initialize dy_pred

    for (int i = 0; i < batch_labels.size(); i++) {
        for (int j = 0; j < kernel_shape["OUTPUT"][1]; j++) {
            if (j == batch_labels[i]) {
                dy_pred[i][j] = 1 - logits[i][j];
            } else {
                dy_pred[i][j] = -1 * logits[i][j];
            }
        }
    }

    std::vector<std::vector<float>> o1_back = o1.backProp(dy_pred);
    printShape(o1_back, "o1_back");
    std::vector<std::vector<float>> a4_back = a4.backProp(o1_back);
    printShape(a4_back, "a4_back");
    std::vector<std::vector<float>> f6_back = f6_layer.back_prop(a4_back);
    printShape(f6_back, "f6_back");
    std::vector<std::vector<float>> a3_back = a3.backProp(f6_back);
    printShape(a3_back, "a3_back");
    std::vector<std::vector<float>> f5_back = f5_layer.back_prop(a3_back);
    printShape(f5_back, "f5_back");
}

std::vector<int> LeNet5::Output_Layer(std::vector<std::vector<float>> X,int outsize){
    int inp = kernel_shape["OUTPUT"][0];
    std::vector<int> Y(outsize,0);
    for (int i = 0; i < outsize; i++) {
        int max_idx = 0;
        float max_val = X[i][max_idx];
        for (int j = 1; j < inp; j++) {
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

void LeNet5::printShape(const std::vector<std::vector<float>>& tensor, const std::string& name) {
    if (tensor.empty()) {
        std::cout << name << " shape: [0]" << std::endl;
        return;
    }
    size_t rows = tensor.size();
    size_t cols = tensor[0].size();
    std::cout << name << " shape: [" << rows << " x " << cols << "]" << std::endl;
}