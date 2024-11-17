#include "dataloader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <cstdlib>
#include <omp.h>


static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
           ((val >> 8)  & 0x0000FF00) |
           ((val << 8)  & 0x00FF0000) |
           ((val << 24) & 0xFF000000);
}

dataloader::dataloader(const std::string& images_path, const std::string& labels_path, int batch_size, bool shuffle)
    : images_path(images_path), labels_path(labels_path), batch_size(batch_size), shuffle(shuffle), current_batch_index(0)
{
    load_data();
    preprocess_data();
    num_batches = images.size() / batch_size;
    pad_images(2);
    if (shuffle) {
        shuffle_data();
    }
}

void dataloader::load_data() {
    std::ifstream image_file(images_path, std::ios::binary);
    if (!image_file.is_open()) {
        std::cerr << "Error opening image file: " << images_path << std::endl;
        exit(1);
    }
    uint32_t magic_number = 0;
    image_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file magic number: " << magic_number << std::endl;
        exit(1);
    }
    uint32_t num_images = 0, num_rows = 0, num_cols = 0;
    image_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = swap_endian(num_images);
    image_file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = swap_endian(num_rows);
    image_file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = swap_endian(num_cols);
    int image_size = num_rows * num_cols;
    images.resize(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        images[i].resize(image_size);
        std::vector<unsigned char> buffer(image_size);
        image_file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        for (int j = 0; j < image_size; ++j) {
            images[i][j] = static_cast<float>(buffer[j]);
        }
    }
    image_file.close();

    std::ifstream label_file(labels_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Error opening label file: " << labels_path << std::endl;
        exit(1);
    }

    magic_number = 0;
    label_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file magic number: " << magic_number << std::endl;
        exit(1);
    }

    // Read label data
    uint32_t num_labels = 0;
    label_file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = swap_endian(num_labels);
    if (num_labels != num_images) {
        std::cerr << "Mismatch between number of images and labels" << std::endl;
        exit(1);
    }

    labels.resize(num_labels);
    std::vector<unsigned char> label_buffer(num_labels);
    label_file.read(reinterpret_cast<char*>(label_buffer.data()), num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        labels[i] = static_cast<int>(label_buffer[i]);
    }
    label_file.close();
}

void dataloader::preprocess_data() {
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel /= 255.0f;
        }
    }
}

std::pair<std::vector<float>, std::vector<int>> dataloader::get_batch() {
    if (current_batch_index >= num_batches) {
        std::cerr << "All batches have been processed. Please reset the dataloader." << std::endl;
        exit(1);
    }

    int start_index = current_batch_index * batch_size;
    int end_index = start_index + batch_size;

    std::vector<float> batch_images;
    std::vector<int> batch_labels;
    int image_size = images[0].size();

    batch_images.reserve(batch_size * image_size);
    batch_labels.reserve(batch_size);

    for (int i = start_index; i < end_index; ++i) {
        batch_images.insert(batch_images.end(), images[i].begin(), images[i].end());
        batch_labels.push_back(labels[i]);
    }

    ++current_batch_index;
    return {batch_images, batch_labels};
}

void dataloader::shuffle_data() {
    std::vector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<float>> shuffled_images(images.size());
    std::vector<int> shuffled_labels(labels.size());
    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_images[i] = images[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    images = std::move(shuffled_images);
    labels = std::move(shuffled_labels);
}

void dataloader::reset() {
    current_batch_index = 0;
    if (shuffle) {
        shuffle_data();
    }
}

void dataloader::pad_images(int padding) {
    int original_size = static_cast<int>(std::sqrt(images[0].size()));
    int new_size = original_size + 2 * padding;
    
    #pragma omp parallel for
    for (auto& image : images) {
        std::vector<float> padded_image(new_size * new_size, 0.0f);

        for (int row = 0; row < original_size; ++row) {
            for (int col = 0; col < original_size; ++col) {
                int original_index = row * original_size + col;
                int padded_index = (row + padding) * new_size + (col + padding);
                padded_image[padded_index] = image[original_index];
            }
        }
        #pragma omp critical
        image = std::move(padded_image);
    }
}

void dataloader::print_images() {
    int num_images = images.size();
    int image_size = images[0].size();

    for (size_t idx = 0; idx < num_images; ++idx) {
        std::cout << "Image " << idx + 1 << ":" << std::endl;
        for (int row = 0; row < image_size; ++row) {
                std::cout << images[idx][row] << " ";
        }
        std::cout << std::endl;
    }
}

