#include <iostream>
#include "lenet5.h"
#include "dataloader.h"
#include<chrono>

int main()
{
        std::string mnist_image_path;
        std::string mnist_label_path;
        // Define the name of the environment variable
        const std::string mnist_images_var = "MNIST_IMAGES_PATH";
        const char* env_var_value_cstr = std::getenv(mnist_images_var.c_str());
        if (env_var_value_cstr) {
                mnist_image_path = std::string(env_var_value_cstr);
                std::cout << "Images path: " << mnist_image_path << std::endl;
        } else {
                // Handle the case where the environment variable is not set
                std::cerr << "Error: Environment variable " << mnist_images_var << " is not set!" << std::endl;
                return 1;
        }
        const std::string mnist_label_var = "MNIST_LABELS_PATH";
        env_var_value_cstr = std::getenv(mnist_label_var.c_str());
        if (env_var_value_cstr) {
                mnist_label_path = std::string(env_var_value_cstr);
                std::cout << "Images path: " << mnist_label_path << std::endl;
        } else {
                // Handle the case where the environment variable is not set
                std::cerr << "Error: Environment variable " << mnist_label_var << " is not set!" << std::endl;
                return 1;
        }
        dataloader dataloader(mnist_image_path,mnist_label_path,128, false);
        LeNet5 lenet;
        int correct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for(int epoch = 1; epoch <= 10; ++epoch) {
                for(int i = 0; i<dataloader.num_batches;i++) {
                        auto x =dataloader.get_batch();
                        std::cout<<i<<": ";
                        int batch_correct  = lenet.Forward_Propagation(x.first, x.second);
                        lenet.Back_Propagation(x.second);
                        correct = batch_correct+correct;
                }
                float accuracy = float(correct)/60000;
                std::cout<<"Accuracy for this epoch: "<<accuracy * 100<<std::endl;
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // Print the duration in milliseconds
                std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
        }
        
}
