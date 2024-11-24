#include "lenet5.h"
#include <iostream>
#include <random>
#include <chrono>

// Constructor
LeNet5::LeNet5()
    : c1_layer(1, CONV1_CHANNELS, 5, 1, 0),  // in_channels, out_channels, kernel_size, stride, padding
      s2_layer(2, 2, 28, CONV1_CHANNELS),    // kernel_size, stride, input_size, num_feature_maps
      c3_layer(CONV1_CHANNELS, CONV3_CHANNELS, 5, 1, 0),
      s4_layer(2, 2, 10, CONV3_CHANNELS),
      f5_layer({FC5_NEURONS, 400}),          // output_size, input_size
      f6_layer({FC6_NEURONS, FC5_NEURONS}),
      o1({OUTPUT_NEURONS, FC6_NEURONS}) {
}

// Forward Propagation
int LeNet5::Forward_Propagation(const std::vector<std::vector<float>>& batch_images,
                              const std::vector<int>& batch_labels) {
    const size_t batch_size = batch_images.size();

    // Forward pass with minimal temporaries
    auto out = c1_layer.forward(batch_images, IMAGE_HEIGHT, IMAGE_WIDTH);
    out = a1.forwardProp(std::move(out));
    out = s2_layer.average_pooling(std::move(out));
    out = a2.forwardProp(std::move(out));
    out = c3_layer.forward(std::move(out), s2_layer.output_image_size, s2_layer.output_image_size);
    out = a3.forwardProp(std::move(out));
    out = s4_layer.average_pooling(std::move(out));
    out = a4.forwardProp(std::move(out));
    out = f5_layer.forward_prop(std::move(out));
    out = a5.forwardProp(std::move(out));
    out = f6_layer.forward_prop(std::move(out));
    out = a6.forwardProp(std::move(out));
    logits = o1.forwardProp(std::move(out));

    // Compute predictions and metrics
    predicted_labels = std::vector<int>(batch_size);
    int correct = 0;

    #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < batch_size; ++i) {
        // Find max probability class
        int max_idx = 0;
        float max_val = logits[i][0];

        #pragma omp simd reduction(max:max_val)
        for (int j = 1; j < OUTPUT_NEURONS; ++j) {
            if (logits[i][j] > max_val) {
                max_idx = j;
                max_val = logits[i][j];
            }
        }

        predicted_labels[i] = max_idx;
        correct += (max_idx == batch_labels[i]);
    }
#ifdef DDEBUG
    // Compute and print metrics
    const float loss = computeLoss(batch_labels);
    std::cout << "Batch Loss: " << loss << " Correct: " << correct << "/" << batch_size
              << " (" << (100.0f * correct / batch_size) << "%)\n";
#endif

    return correct;
}

void LeNet5::Back_Propagation(const std::vector<int>& batch_labels) {
    const size_t batch_size = batch_labels.size();

    // Compute gradients for cross-entropy loss
    std::vector<std::vector<float>> gradients(batch_size,
        std::vector<float>(OUTPUT_NEURONS));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < batch_size; ++i) {
        for (int j = 0; j < OUTPUT_NEURONS; ++j) {
            gradients[i][j] = logits[i][j];
            if (j == batch_labels[i]) {
                gradients[i][j] -= 1.0f;
            }
        }
    }

    // Backward pass with minimal temporaries
    auto grad = o1.backProp(std::move(gradients));
    grad = a6.backProp(std::move(grad));
    grad = f6_layer.back_prop(std::move(grad));
    grad = a5.backProp(std::move(grad));
    grad = f5_layer.back_prop(std::move(grad));
    grad = a4.backProp(std::move(grad));
    grad = s4_layer.backward(std::move(grad));
    grad = a3.backProp(std::move(grad));
    grad = c3_layer.backward(std::move(grad));
    grad = a2.backProp(std::move(grad));
    grad = s2_layer.backward(std::move(grad));
    grad = a1.backProp(std::move(grad));
    grad = c1_layer.backward(std::move(grad));
}

float LeNet5::computeLoss(const std::vector<int>& batch_labels) const {
    float total_loss = 0.0f;
    const size_t batch_size = batch_labels.size();

    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        total_loss += -std::log(logits[i][batch_labels[i]]);
    }

    return total_loss / batch_size;
}