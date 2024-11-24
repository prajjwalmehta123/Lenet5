// main.cpp
#include <iostream>
#include <chrono>
#include <queue>
#include "lenet5.h"
#include "dataloader.h"

class TrainingPipeline {
private:
    LeNet5 model;
    dataloader& train_loader;
    const size_t num_threads;

    struct BatchMetrics {
        int correct;
        float loss;
    };

public:
    TrainingPipeline(dataloader& loader, size_t threads = 4)
        : train_loader(loader), num_threads(threads) {}

    void train(int num_epochs) {
        for (int epoch = 1; epoch <= num_epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            int total_correct = 0;
            float total_loss = 0.0f;

            // Process all batches
            for (int batch = 0; batch < train_loader.num_batches; ++batch) {
                // Get next batch
                auto [images, labels] = train_loader.get_batch();

                // Process batch
                int batch_correct = model.Forward_Propagation(images, labels);
                model.Back_Propagation(labels);
                total_correct += batch_correct;

                // Print progress
                if ((batch + 1) % 19 == 0) {
                    float progress = (batch + 1.0f) / train_loader.num_batches * 100;
                    float current_accuracy = static_cast<float>(total_correct) /
                        ((batch + 1) * train_loader.num_batches) * 100;

                    std::printf("\rEpoch %d: %.1f%% [%d/%d] - Accuracy: %.2f%%",
                              epoch, progress, batch + 1, train_loader.num_batches,
                              current_accuracy);
                    std::fflush(stdout);
                }
            }

            // Compute epoch metrics
            float accuracy = static_cast<float>(total_correct) /
                (train_loader.num_batches * train_loader.num_batches) * 100;

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

            std::printf("\nEpoch %d completed in %.2fs, Accuracy: %.2f%%\n",
                       epoch, duration.count() / 1000.0f, accuracy);

            // Reset for next epoch
            train_loader.reset();
        }
    }
};

int main() {
    // Get MNIST paths from environment variables
    const char* mnist_images = std::getenv("MNIST_IMAGES_PATH");
    const char* mnist_labels = std::getenv("MNIST_LABELS_PATH");

    if (!mnist_images || !mnist_labels) {
        std::cerr << "Error: MNIST path environment variables not set!\n";
        return 1;
    }

    try {
        const int BATCH_SIZE = 128;
        dataloader train_loader(mnist_images, mnist_labels, BATCH_SIZE, true);
        TrainingPipeline pipeline(train_loader);
        auto total_start = std::chrono::high_resolution_clock::now();
        pipeline.train(10);
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
        std::cout << "\nTotal training time: " << total_duration.count() << "s\n";

    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}