# LeNet-5 Implementation in C++

This repository contains a modern C++ implementation of the LeNet-5 convolutional neural network architecture, designed for MNIST digit classification. The implementation features both CPU (OpenMP) and GPU (CUDA) acceleration options.

## Features

- Complete LeNet-5 architecture implementation
- MNIST dataset support
- OpenMP parallel processing for CPU acceleration
- Optional CUDA support for GPU acceleration
- Batch processing capability
- Adam optimizer implementation
- Modular design with separate layer implementations

## Prerequisites

### Required:
- C++17 compatible compiler
- CMake (minimum version 3.16)
- OpenMP

### Optional:
- CUDA Toolkit (for GPU acceleration)
- Compatible NVIDIA GPU

## Building the Project

1. Clone the repository:
```bash
git clone [repository-url]
cd lenet5-implementation
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure with CMake:

For CPU-only build:
```bash
cmake ..
```

For GPU-enabled build:
```bash
cmake -DUSE_CUDA=ON ..
```

4. Build the project:
```bash
make
```

## Usage

1. Set up environment variables for MNIST dataset paths:
```bash
export MNIST_IMAGES_PATH=/path/to/train-images-idx3-ubyte
export MNIST_LABELS_PATH=/path/to/train-labels-idx1-ubyte
export MNIST_TEST_IMAGES_PATH=/path/to/t10k-images-idx3-ubyte
export MNIST_TEST_LABELS_PATH=/path/to/t10k-labels-idx1-ubyte
```

2. Run the executable:
```bash
./lenet5
```

## Dataset

The implementation uses the MNIST dataset. You can download it from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download).

The dataset should be in the IDX format:
- Training images: train-images-idx3-ubyte
- Training labels: train-labels-idx1-ubyte
- Test images: t10k-images-idx3-ubyte
- Test labels: t10k-labels-idx1-ubyte

## Implementation Details

### Network Architecture
- Input Layer: 32x32 grayscale images (padded from 28x28)
- C1: Convolutional layer (6 feature maps, 5x5 kernels)
- S2: Average pooling layer (2x2)
- C3: Convolutional layer (16 feature maps, 5x5 kernels)
- S4: Average pooling layer (2x2)
- F5: Fully connected layer (120 neurons)
- F6: Fully connected layer (84 neurons)
- Output: Fully connected layer (10 neurons)

### Optimization
- Adam optimizer with configurable parameters
- ReLU activation functions
- OpenMP parallelization for CPU
- Optional CUDA acceleration for GPU

## Performance

- CPU Version: Utilizes OpenMP for parallel processing
- GPU Version: Supports CUDA acceleration (requires `-DUSE_CUDA=ON` during build)
- Batch sizes:
  - CPU: Default batch size of 64
  - GPU: Default batch size of 256