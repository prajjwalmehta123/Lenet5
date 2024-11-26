#include "model.h"
#include <iostream>
#include <chrono>
#include <iomanip>

Model::Model() {}

void Model::train(dataloader& train_loader, int num_epochs, int print_every) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for(int epoch = 1; epoch <= num_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch << "/" << num_epochs << std::endl;
        
        float epoch_loss = 0.0f;
        int total_correct = 0;
        int total_samples = 0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        for(int step = 0; step < train_loader.num_batches; step++) {
            auto batch = train_loader.get_batch();
            
            // Forward pass and compute loss
            network.Forward_Propagation(batch.first, batch.second);
            float step_loss = network.compute_loss(batch.second);
            int step_correct = network.compute_accuracy(batch.second);
            
            // Backward pass
            network.Back_Propagation(batch.second);
            
            // Accumulate metrics
            epoch_loss += step_loss;
            total_correct += step_correct;
            total_samples += batch.first.size();
            
            // Print progress
            if ((step + 1) % print_every == 0) {
                float current_loss = epoch_loss / (step + 1);
                float current_accuracy = 100.0f * total_correct / total_samples;
                
                std::cout << "Step " << std::setw(5) << step + 1 << "/"
                          << std::setw(5) << train_loader.num_batches
                          << " - Loss: " << std::fixed << std::setprecision(4) << current_loss
                          << " - Accuracy: " << std::setprecision(2) << current_accuracy << "%"
                          << std::endl;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        // Compute epoch metrics
        float final_epoch_loss = epoch_loss / train_loader.num_batches;
        float final_epoch_accuracy = 100.0f * total_correct / total_samples;
        
        // Store metrics
        TrainingMetrics metrics{final_epoch_loss, final_epoch_accuracy, epoch_duration};
        training_history.push_back(metrics);
        
        std::cout << "\nEpoch " << epoch << " Summary:"
                  << "\nLoss: " << std::fixed << std::setprecision(4) << final_epoch_loss
                  << "\nAccuracy: " << std::setprecision(2) << final_epoch_accuracy << "%"
                  << "\nTime: " << epoch_duration.count() / 1000.0 << " seconds"
                  << std::endl;
        
        train_loader.reset();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "\nTotal training time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
}

float Model::test(dataloader& test_loader) {
    std::cout << "\nEvaluating model on test set..." << std::endl;
    
    int total_correct = 0;
    int total_samples = 0;
    float total_loss = 0.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int step = 0; step < test_loader.num_batches; step++) {
        auto batch = test_loader.get_batch();
        
        // Forward pass only
        network.Forward_Propagation(batch.first, batch.second);
        
        // Accumulate metrics
        total_correct += network.compute_accuracy(batch.second);
        total_loss += network.compute_loss(batch.second);
        total_samples += batch.first.size();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float accuracy = 100.0f * total_correct / total_samples;
    float avg_loss = total_loss / test_loader.num_batches;
    
    std::cout << "Test Results:"
              << "\nAccuracy: " << std::fixed << std::setprecision(2) << accuracy << "%"
              << "\nAverage Loss: " << std::setprecision(4) << avg_loss
              << "\nTime: " << duration.count() / 1000.0 << " seconds"
              << std::endl;
    
    return accuracy;
}

void Model::save(const std::string& filepath) {
    std::ofstream outfile(filepath, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
    try {
        network.saveModel("lenet5_model.bin");
        std::cout << "Model saved successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }  
    std::cout << "Model saved to: " << filepath << std::endl;
    outfile.close();
}

void Model::load(const std::string& filepath) {
    std::ifstream infile(filepath, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading: " + filepath);
    }
    try {
        network.loadModel("lenet5_model.bin");
        //network = loaded_model;
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }    
    std::cout << "Model loaded from: " << filepath << std::endl;
    infile.close();
}