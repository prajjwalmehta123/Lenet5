#include <iostream>
#include <vector>
#include <iomanip> // For setting precision
#include <cassert>
#include "FCLayer.h"

// Function to print a 2D vector
void print2DVector(const std::vector<std::vector<float>>& vec) {
    for (const auto& row : vec) {
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    // Define the test input batch
    std::vector<std::vector<float>> inputBatch = {
        {1.0f, 2.0f, 3.0f}, // Sample 1
        {4.0f, 5.0f, 6.0f}  // Sample 2
    };

    // Define layer dimensions
    int inputSize = 3;
    int outputSize = 2;

    // Create an instance of FCLayer
    FCLayer fcLayer({outputSize, inputSize}, "manual");

    // Manually set weights and biases
    fcLayer.weight = {
        {0.1f, 0.2f, 0.3f}, // Weights for neuron 0
        {0.4f, 0.5f, 0.6f}  // Weights for neuron 1
    };
    fcLayer.bias = {0.5f, 0.6f}; // Biases for neurons

    // Perform forward propagation
    std::vector<std::vector<float>> outputBatch = fcLayer.forward_prop(inputBatch);

    // Print the outputs
    std::cout << "Forward Propagation Output:\n";
    print2DVector(outputBatch);

    // Define the gradient from the next layer (dZ)
    std::vector<std::vector<float>> dZ = {
        {1.0f, 1.5f}, // Gradient for Sample 1
        {2.0f, 2.5f}  // Gradient for Sample 2
    };

    // Perform backward propagation
    std::vector<std::vector<float>> dA_prev = fcLayer.back_prop(dZ);

    // Print the gradients w.r.t. inputs
    std::cout << "\nBackward Propagation Output (Gradients w.r.t Input):\n";
    print2DVector(dA_prev);

    // Expected gradients w.r.t inputs
    std::vector<std::vector<float>> expected_dA_prev = {
        {0.7f, 0.95f, 1.2f},
        {1.2f, 1.65f, 2.1f}
    };

    // Verify dA_prev
    for (size_t i = 0; i < dA_prev.size(); ++i) {
        for (size_t j = 0; j < dA_prev[0].size(); ++j) {
            assert(fabs(dA_prev[i][j] - expected_dA_prev[i][j]) < 1e-4);
        }
    }

    // Expected gradients w.r.t weights
    std::vector<std::vector<float>> expected_dW = {
        {4.5f, 6.0f, 7.5f},   // For neuron 0
        {5.75f, 7.75f, 9.75f} // For neuron 1
    };

    // Expected gradients w.r.t biases
    std::vector<float> expected_db = {1.5f, 2.0f};

    // Since weights and biases are private, you need to modify your class to store gradients for testing
    // Assuming you have public members or methods to access dW and db
    // For the purpose of this test, let's assume we have access to dW and db

    // Print expected gradients
    std::cout << "\nExpected Gradients w.r.t Weights (dW):\n";
    print2DVector(expected_dW);

    std::cout << "\nExpected Gradients w.r.t Biases (db):\n";
    for (float val : expected_db) {
        std::cout << std::fixed << std::setprecision(4) << val << "\t";
    }
    std::cout << "\n";

    // Verify the gradients computed during backpropagation
    // Assuming your FCLayer class stores the computed gradients in member variables
    // Let's access them (you may need to adjust your class to expose them for testing)

    // For demonstration, let's assume fcLayer.dW and fcLayer.db exist and are accessible
    // Replace these with actual code to access gradients in your implementation

    
    // Print computed gradients
     std::cout << "\nComputed Gradients w.r.t Weights (dW):\n";
        print2DVector(fcLayer.dW);

        std::cout << "\nComputed Gradients w.r.t Biases (db):\n";
        for (float val : fcLayer.db) {
        std::cout << std::fixed << std::setprecision(4) << val << "\t";
        }
        std::cout << "\n";

    // Verify dW and db
    // Verify dW and db
for (size_t i = 0; i < fcLayer.dW.size(); ++i) {
    for (size_t j = 0; j < fcLayer.dW[0].size(); ++j) {
        assert(fabs(fcLayer.dW[i][j] - expected_dW[i][j]) < 1e-4);
    }
}

for (size_t i = 0; i < fcLayer.db.size(); ++i) {
    assert(fabs(fcLayer.db[i] - expected_db[i]) < 1e-4);
}
    

    std::cout << "Backward propagation test completed successfully.\n";

    return 0;
}