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

void LeNet5::saveModel(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }

    // Save conv layers
    writeConvLayer<4>(file, c1_layer.getWeights(), c1_layer.getBiases());
    writeConvLayer<4>(file, c3_layer.getWeights(), c3_layer.getBiases());

    // Save fully connected layers
    writeLayer(file, f5_layer.getWeights(), f5_layer.getBiases());
    writeLayer(file, f6_layer.getWeights(), f6_layer.getBiases());
    writeLayer(file, o1.getWeights(), o1.getBiases());

    file.close();
}

void LeNet5::loadModel(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }

    // Load conv layers
    auto c1_weights = c1_layer.getWeights();
    auto c1_biases = c1_layer.getBiases();
    readConvLayer<4>(file, c1_weights, c1_biases);

    auto c3_weights = c3_layer.getWeights();
    auto c3_biases = c3_layer.getBiases();
    readConvLayer<4>(file, c3_weights, c3_biases);

    // Load fully connected layers
    auto f5_weights = f5_layer.getWeights();
    auto f5_biases = f5_layer.getBiases();
    readLayer(file, f5_weights, f5_biases);

    auto f6_weights = f6_layer.getWeights();
    auto f6_biases = f6_layer.getBiases();
    readLayer(file, f6_weights, f6_biases);

    auto o1_weights = o1.getWeights();
    auto o1_biases = o1.getBiases();
    readLayer(file, o1_weights, o1_biases);

    file.close();
}

void LeNet5::writeLayer(std::ofstream& file, const std::vector<std::vector<float>>& weights,
                       const std::vector<float>& biases) const {
    // Write dimensions
    size_t rows = weights.size();
    size_t cols = weights[0].size();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Write weights
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }

    // Write biases
    size_t biasSize = biases.size();
    file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
    file.write(reinterpret_cast<const char*>(biases.data()), biasSize * sizeof(float));
}

void LeNet5::readLayer(std::ifstream& file, std::vector<std::vector<float>>& weights,
                      std::vector<float>& biases) {
    // Read dimensions
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Read weights
    weights.resize(rows, std::vector<float>(cols));
    for (auto& row : weights) {
        file.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }

    // Read biases
    size_t biasSize;
    file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
    biases.resize(biasSize);
    file.read(reinterpret_cast<char*>(biases.data()), biasSize * sizeof(float));
}

template<size_t N>
void LeNet5::writeConvLayer(std::ofstream& file,
                           const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                           const std::vector<float>& biases) const {
    // Write dimensions
    std::array<size_t, N> dims;
    dims[0] = weights.size();
    dims[1] = weights[0].size();
    dims[2] = weights[0][0].size();
    dims[3] = weights[0][0][0].size();
    
    file.write(reinterpret_cast<const char*>(dims.data()), N * sizeof(size_t));

    // Write weights
    for (const auto& d1 : weights) {
        for (const auto& d2 : d1) {
            for (const auto& d3 : d2) {
                file.write(reinterpret_cast<const char*>(d3.data()), d3.size() * sizeof(float));
            }
        }
    }

    // Write biases
    size_t biasSize = biases.size();
    file.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
    file.write(reinterpret_cast<const char*>(biases.data()), biasSize * sizeof(float));
}

template<size_t N>
void LeNet5::readConvLayer(std::ifstream& file,
                          std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
                          std::vector<float>& biases) {
    // Read dimensions
    std::array<size_t, N> dims;
    file.read(reinterpret_cast<char*>(dims.data()), N * sizeof(size_t));

    // Resize weights
    weights.resize(dims[0]);
    for (auto& d1 : weights) {
        d1.resize(dims[1]);
        for (auto& d2 : d1) {
            d2.resize(dims[2]);
            for (auto& d3 : d2) {
                d3.resize(dims[3]);
            }
        }
    }

    // Read weights
    for (auto& d1 : weights) {
        for (auto& d2 : d1) {
            for (auto& d3 : d2) {
                file.read(reinterpret_cast<char*>(d3.data()), d3.size() * sizeof(float));
            }
        }
    }

    // Read biases
    size_t biasSize;
    file.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
    biases.resize(biasSize);
    file.read(reinterpret_cast<char*>(biases.data()), biasSize * sizeof(float));
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