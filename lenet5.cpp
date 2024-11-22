#include "lenet5.h"
#include <iostream>
#include <random>
#include <chrono>

// Constructor
LeNet5::LeNet5(){
    /*
    for (const auto& pair : kernel_shape) {
        std::cout << pair.first << ": ";
        for (const auto& val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    */
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
int LeNet5::Forward_Propagation(std::vector<std::vector<float>> batch_images, std::vector<int>batch_labels) {
    std::vector<std::vector<float>> out = c1_layer.forward(batch_images, imageHeight, imageWidth);
    printShape(out, "c1_out");
    out = a1.forwardProp(out);
    printShape(out, "a1_out");
    out = s2_layer.average_pooling(out);
    printShape(out, "s2_out");
    out = a2.forwardProp(out);
    printShape(out, "a2_out");
    out = c3_layer.forward(out, s2_layer.output_image_size, s2_layer.output_image_size);
    printShape(out, "c3_out");
    out = a3.forwardProp(out);
    printShape(out, "a3_out");
    out = s4_layer.average_pooling(out);
    printShape(out, "s4_out");
    out = a4.forwardProp(out);
    printShape(out, "a4_out");
    out = f5_layer.forward_prop(out);
    printShape(out, "f5_out");
    out = a5.forwardProp(out);
    printShape(out, "a5_out");
    out = f6_layer.forward_prop(out);
    printShape(out, "f6_out");
    out = a6.forwardProp(out);
    printShape(out, "a6_out");
    logits = o1.forwardProp(out);
    printShape(logits, "logits");
    labels = Output_Layer(logits,batch_images.size());
    //std::cout<<"Label size: "<<labels.size();
    int correct = 0;
    for(int i = 0; i< labels.size();i++){
        if (labels[i] == batch_labels[i]) {
            correct++;
        }
    }
    float total_loss = 0.0f;
    for (size_t i = 0; i < batch_labels.size(); ++i) {
        int correct_label = batch_labels[i];
        float prob = logits[i][correct_label]; // Softmax probability for the correct class
        total_loss += -std::log(prob); // Cross-entropy loss
    }
    total_loss /= batch_labels.size(); // Average loss for the batch
    std::cout<<"Average Loss For this batch:  "<<total_loss<<" Correct This Batch: "<<correct<<std::endl;

    return correct;
}

// Back Propagation
void LeNet5::Back_Propagation(std::vector<int>batch_labels) {
    std::vector<std::vector<float>> dy_pred(batch_labels.size(), std::vector<float>(kernel_shape["OUTPUT"][0], 0));
    for (size_t i = 0; i < batch_labels.size(); ++i) {
        for (size_t j = 0; j < logits[i].size(); ++j) {
            dy_pred[i][j] = logits[i][j]; // Copy softmax probabilities
            if (j == batch_labels[i]) {
                dy_pred[i][j] -= 1.0f; // Subtract 1 for the correct class
            }
        }
    }
    std::vector<std::vector<float>> back_out = o1.backProp(dy_pred);
    printShape(back_out, "o1_back");
    back_out = a6.backProp(back_out);
    printShape(back_out, "a6_back");
    back_out = f6_layer.back_prop(back_out);
    printShape(back_out, "f6_back");
    back_out = a5.backProp(back_out);
    printShape(back_out, "a5_back");
    back_out = f5_layer.back_prop(back_out);
    printShape(back_out, "f5_back");
    back_out = a4.backProp(back_out);
    printShape(back_out, "a4_back");
    back_out = s4_layer.backward(back_out);
    printShape(back_out, "s4_back");
    back_out = a3.backProp(back_out);
    printShape(back_out, "a3_back");
    back_out = c3_layer.backward(back_out);
    printShape(back_out, "c3_back");
    back_out = a2.backProp(back_out);
    printShape(back_out, "a2_back");
    back_out = s2_layer.backward(back_out);
    printShape(back_out, "s2_back");
    back_out = a1.backProp(back_out);
    printShape(back_out, "a1_back");
    back_out = c1_layer.backward(back_out);
    printShape(back_out, "c1_back");

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
    return;
    if (tensor.empty()) {
         std::cout << name << " shape: [0]" << std::endl;
         return;
     }
    size_t rows = tensor.size();
    size_t cols = tensor[0].size();
    std::cout << name << " shape: [" << rows << " x " << cols << "]" << std::endl;
}