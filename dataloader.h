#ifndef DATALOADER_H
#define DATALOADER_H
#include <vector>

class dataloader {
private:
    std::vector<std::vector<float>> images;     // Store images as flattened vectors
    std::vector<int> labels;                    // Store labels as integer vectors
    int batch_size{};                             // Number of samples per batch
    int num_batches{};                            // Total number of batches per epoch
    int current_batch_index{};                    // Current batch index in the epoch
    int padding{};
    bool shuffle{};                               // Flag to shuffle dataset before each epoch
    std::string images_path;
    std::string labels_path;

public:
    dataloader(const std::string& images_path, const std::string& labels_path, int batch_size, bool shuffle = true);
    void load_data();
    void preprocess_data();
    std::pair<std::vector<float>, std::vector<int>> get_batch();
    void shuffle_data();
    void reset();
    void pad_images(int padding);
    void print_images();
};


#endif //DATALOADER_H
