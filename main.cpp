#include <iostream>

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
        auto x =dataloader.get_batch();

        int inputChannels = 1;
        int outputChannels = 6;
        int kernelSize = 5;
        int stride = 1;
        int padding = 0;
        ConvolutionLayer convLayer(inputChannels, outputChannels, kernelSize, stride, padding);
        std::vector<std::vector<float>> inputBatch = x.first;
        int imageHeight = 32;  // 32x32 images (already padded)
        int imageWidth = 32;
        std::vector<std::vector<float>> outputBatch = convLayer.forward(inputBatch, imageHeight, imageWidth);
        int outputHeight = ConvolutionLayer::calculateOutputSize(imageHeight, kernelSize, stride, padding);
        int outputWidth = ConvolutionLayer::calculateOutputSize(imageWidth, kernelSize, stride, padding);
        outputChannels = convLayer.outputChannels;
        int outputSizePerImage = outputHeight * outputWidth * outputChannels;

        std::cout << "Batch size: " << outputBatch.size() << std::endl;
        std::cout << "Output size per image: " << outputBatch[0].size() << std::endl;
        std::cout << "Expected output size per image: " << outputSizePerImage << std::endl;


        std::cout<<'x'<<std::endl;
}
