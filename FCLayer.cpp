#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include "FCLayer.h"

// Default Constructor
FCLayer::FCLayer() {
    AdamOptimizer adam(0.01, 0.9, 0.99, 1e-8);
}

// Constructor
FCLayer::FCLayer(const std::pair<int, int>& weight_shape, const std::string& init_mode) {
    int rows = weight_shape.first;
    int cols = weight_shape.second;
    // Initialize weights and biases
    std::tie(weight, bias) = initialize(rows, cols);
}

// std::vector<std::vector<float>> FCLayer::transpose(const std::vector<std::vector<float>>& matrix) {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             transposed[j][i] = matrix[i][j];
//         }
//     }
//     return transposed;
// }

// Forward Propagation
std::vector<std::vector<float>> FCLayer::forward_prop(const std::vector<std::vector<float>>& input_array) {
    this->input_array = input_array; // Cache input for backpropagation

    int batch_size = input_array.size();
    int output_size = weight.size();
    int input_size = input_array[0].size();
    // auto weight_T = transpose(weight);

    std::vector<std::vector<float>> output(batch_size, std::vector<float>(output_size, 0.0f));

    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < output_size; ++j) {
    //         float sum = bias[j];
    //         #pragma omp simd reduction(+:sum)
    //         for (int k = 0; k < input_size; ++k) {
    //             sum += input_array[i][k] * weight_T[k][j];
    //         }
    //         output[i][j] = sum;
    //     }
    // }
    int block_size = 16; // Choose based on cache size
    #pragma omp parallel for collapse(2) // Parallelize across the first two loops
    for (int ii = 0; ii < batch_size; ii += block_size) {
        for (int jj = 0; jj < output_size; jj += block_size) {
            for (int kk = 0; kk < input_size; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, batch_size); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, output_size); ++j) {
                        float sum = 0.0f; // Initialize sum locally to avoid race conditions
                        for (int k = kk; k < std::min(kk + block_size, input_size); ++k) {
                            sum += input_array[i][k] * weight[j][k];
                        }
                        #pragma omp atomic // Ensure thread safety for shared output
                        output[i][j] += sum + bias[j];
                    }
                }
            }
        }
    }
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < output_size; ++j) {
    //         float sum = bias[j];
    //         #pragma omp simd reduction(+:sum)
    //         for (int k = 0; k < input_size; ++k) {
    //             sum += input_array[i][k] * weight[j][k];
    //         }
    //         output[i][j] = sum;
    //     }
    // }
    return output;
}

// Backward Propagation
std::vector<std::vector<float>> FCLayer::back_prop(const std::vector<std::vector<float>>& dZ) {
    int batch_size = dZ.size();
    int output_size = weight.size();       // Number of output neurons
    int input_size = weight[0].size();     // Number of input features

    // Initialize gradients
    dA_prev.assign(batch_size, std::vector<float>(input_size, 0.0f));
    dW.assign(output_size, std::vector<float>(input_size, 0.0f));
    db.assign(output_size, 0.0f);

    #pragma omp parallel
    {
        // Thread-local storage for partial gradient accumulations
        std::vector<std::vector<float>> thread_dW(output_size, std::vector<float>(input_size, 0.0f));
        std::vector<float> thread_db(output_size, 0.0f);
        std::vector<std::vector<float>> thread_dA_prev(batch_size, std::vector<float>(input_size, 0.0f));

        #pragma omp for
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                thread_db[j] += dZ[i][j];
                #pragma omp simd
                for (int k = 0; k < input_size; ++k) {
                    thread_dW[j][k] += dZ[i][j] * input_array[i][k];
                    thread_dA_prev[i][k] += dZ[i][j] * weight[j][k];
                }
            }
        }

        // Merge thread-local results into global storage
        #pragma omp critical
        {
            for (int j = 0; j < output_size; ++j) {
                db[j] += thread_db[j];
                #pragma omp simd
                for (int k = 0; k < input_size; ++k) {
                    dW[j][k] += thread_dW[j][k];
                }
            }

            for (int i = 0; i < batch_size; ++i) {
                #pragma omp simd
                for (int k = 0; k < input_size; ++k) {
                    dA_prev[i][k] += thread_dA_prev[i][k];
                }
            }
        }
    }

    // Average gradients over the batch
    float scale = 1.0f / batch_size;
    #pragma omp parallel for
    for (int j = 0; j < output_size; ++j) {
        db[j] *= scale;
        #pragma omp simd
        for (int k = 0; k < input_size; ++k) {
            dW[j][k] *= scale;
        }
    }

    // Update weights and biases using Adam optimizer
    adam.update_weight(weight, dW);
    adam.update_bias(bias, db);

    return dA_prev;
}

// Initialize weights and biases
std::pair<std::vector<std::vector<float>>, std::vector<float>> FCLayer::initialize(int rows, int cols) {
    std::vector<std::vector<float>> w(rows, std::vector<float>(cols, 0.0f));
    std::vector<float> b(cols, 0.01f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 0.1f);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            w[i][j] = d(gen);
        }
    }
    return {w, b};
}