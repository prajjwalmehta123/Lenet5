// Model.h
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <fstream>
#include<chrono>
#include "lenet5.h"
#include "dataloader.h"

class Model {
public:
    Model();
    void train(dataloader& train_loader, int num_epochs, int print_every = 50);
    float test(dataloader& test_loader);
    void save(const std::string& filepath);
    void load(const std::string& filepath);

    struct TrainingMetrics {
        float loss;
        float accuracy;
        std::chrono::milliseconds duration;
    };

private:
    LeNet5 network;
    std::vector<TrainingMetrics> training_history;
};

#endif // MODEL_H