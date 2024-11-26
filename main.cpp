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
        std::string mnist_test_image_path;
        std::string mnist_test_label_path;
        // Define the name of the environment variable
        const std::string mnist_test_images_var = "MNIST_TEST_IMAGES_PATH";
        const char* test_env_var_value_cstr = std::getenv(mnist_test_images_var.c_str());
        if (test_env_var_value_cstr) {
                mnist_test_image_path = std::string(test_env_var_value_cstr);
                std::cout << "Test Images path: " << mnist_test_image_path << std::endl;
        } else {
                // Handle the case where the environment variable is not set
                std::cerr << "Error: Test Environment variable " << mnist_test_images_var << " is not set!" << std::endl;
                return 1;
        }
        const std::string mnist_test_label_var = "MNIST_TEST_LABELS_PATH";
        test_env_var_value_cstr = std::getenv(mnist_test_label_var.c_str());
        if (test_env_var_value_cstr) {
                mnist_test_label_path = std::string(test_env_var_value_cstr);
                std::cout << "Test Images path: " << mnist_test_label_path << std::endl;
        } else {
                // Handle the case where the environment variable is not set
                std::cerr << "Error: Test Environment variable " << mnist_test_label_var << " is not set!" << std::endl;
                return 1;
        }
        #ifdef USE_CUDA
        dataloader dataloader_train(mnist_image_path,mnist_label_path,256, false);
        #else
         dataloader dataloader_train(mnist_image_path,mnist_label_path,64, false);
        #endif
       
        LeNet5 lenet;
        
        auto start = std::chrono::high_resolution_clock::now();
        for(int epoch = 1; epoch <= 10; ++epoch) {
                for(int i = 0; i<dataloader_train.num_batches;i++) {
                        auto x =dataloader_train.get_batch();
                        lenet.Forward_Propagation(x.first, x.second);
                        lenet.Back_Propagation(x.second);
                        if (i % 50 == 0) {
                                std::cout << "Epoch " << epoch << ", Step " << i 
                                        << " - Loss: " << lenet.compute_loss(x.second) << std::endl;
                        }
                        // break;
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Time taken: " << duration.count()/1000 << " seconds" << std::endl;
                dataloader_train.reset();
                break;
        }
        int correct = 0;
        dataloader dataloader_test(mnist_test_image_path,mnist_test_label_path,16, false);
        for(int i = 0; i<dataloader_test.num_batches;i++) {
                        auto x =dataloader_test.get_batch();
                        lenet.Forward_Propagation(x.first, x.second);
                        correct+=lenet.compute_accuracy(x.second);
                }
        float accuracy = float(correct)/10000;
        std::cout<<"Accuracy for Test : "<<accuracy * 100<<std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken: " << duration.count()/1000 << " seconds" << std::endl;
}

