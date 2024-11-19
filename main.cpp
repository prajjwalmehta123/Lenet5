#include <iostream>
#include "lenet5.h"
#include "conv.h"
#include "dataloader.h"

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
        dataloader dataloader(mnist_image_path,mnist_label_path,32);
        LeNet5 lenet;
        for(int i = 0; i<dataloader.num_batches;i++) {
                auto x =dataloader.get_batch();
                lenet.Forward_Propagation(x.first, x.second);
                //break;
        }

}
