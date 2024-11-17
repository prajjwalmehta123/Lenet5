#ifndef DATALOADER_H
#define DATALOADER_H
#include <vector>

using namespace std;
class dataloader {
private:
    vector<vector<float>> images;     // Store images as flattened vectors
    vector<int> labels;                    // Store labels as integer vectors
    int batch_size{};                             // Number of samples per batch
    int num_batches{};                            // Total number of batches per epoch
    int current_batch_index{};                    // Current batch index in the epoch
    int padding{};
    bool shuffle{};                               // Flag to shuffle dataset before each epoch
    string images_path;
    string labels_path;

public:
    dataloader(const string& images_path, const string& labels_path, int batch_size, bool shuffle = true);
    void load_data();
    void preprocess_data();
    pair<vector<vector<float>>, vector<int>> get_batch();
    void shuffle_data();
    void reset();
    void print_images();
};


#endif //DATALOADER_H
