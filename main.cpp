#include <iostream>
#include <string>
#include "model.h"
#include "dataloader.h"

std::string getEnvVar(const std::string& var_name) {
    const char* val = std::getenv(var_name.c_str());
    if (!val) {
        throw std::runtime_error("Environment variable " + var_name + " is not set!");
    }
    return std::string(val);
}

int main() {
    try {
        // Get environment variables
        std::string mnist_image_path = getEnvVar("MNIST_IMAGES_PATH");
        std::string mnist_label_path = getEnvVar("MNIST_LABELS_PATH");
        std::string mnist_test_image_path = getEnvVar("MNIST_TEST_IMAGES_PATH");
        std::string mnist_test_label_path = getEnvVar("MNIST_TEST_LABELS_PATH");
        
        // Initialize dataloaders
        #ifdef USE_CUDA
            dataloader train_loader(mnist_image_path, mnist_label_path, 256, true);
        #else
            dataloader train_loader(mnist_image_path, mnist_label_path, 128, true);
        #endif
        dataloader test_loader(mnist_test_image_path, mnist_test_label_path, 16, false);
        
        Model model;
        
        std::cout << "Starting training..." << std::endl;
        model.train(train_loader, 10); // 10 epochs
        // Testing
        std::cout << "\nStarting evaluation..." << std::endl;
        float test_accuracy = model.test(test_loader);
        
        // Save the model
        model.save("lenet5_model.bin");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}