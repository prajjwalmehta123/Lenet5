cmake_minimum_required(VERSION 3.16)
project(lenet5 LANGUAGES CXX CUDA)

option(USE_CUDA "Enable GPU support using CUDA" OFF)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    main.cpp
    lenet5.cpp
    dataloader.cpp
    FCLayer.cpp
    conv.cpp
    subsampling.cpp
    out.cpp
    activation.cpp
    adam.cpp
)

# GPU-specific settings
if(USE_CUDA)
    enable_language(CUDA)
    
    # Explicitly set paths for Conda CUDA installation
    set(CONDA_CUDA_TOOLKIT_PATH "$ENV{CONDA_PREFIX}")
    set(CMAKE_CUDA_COMPILER "${CONDA_CUDA_TOOLKIT_PATH}/bin/nvcc")
    include_directories("${CONDA_CUDA_TOOLKIT_PATH}/include")
    link_directories("${CONDA_CUDA_TOOLKIT_PATH}/lib" "${CONDA_CUDA_TOOLKIT_PATH}/lib64")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -arch=sm_61 \
        -O3 \
        --use_fast_math")
    # Add CUDA source
    list(APPEND SOURCES conv_gpu.cu cuda_utils.cu fc_gpu.cu subsampling_gpu.cu activation_gpu.cu output_gpu.cu)
    
    # Set specific architecture for Tesla P4 (Compute Capability 6.1)
    set(CMAKE_CUDA_ARCHITECTURES 61)
    
    # Set CUDA flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    
    message(STATUS "CUDA support enabled for Tesla P4")
    message(STATUS "CUDA Toolkit Path: ${CONDA_CUDA_TOOLKIT_PATH}")
endif()

# Define the executable
add_executable(lenet5 ${SOURCES})

# Apply GPU-specific compile options and flags
if(USE_CUDA)
    #target_compile_features(lenet5 PUBLIC cuda_std_17)
    target_compile_definitions(lenet5 PUBLIC USE_CUDA)
    #target_link_libraries(lenet5 PUBLIC cuda cudart)

    
    set_target_properties(lenet5 PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON)
endif()

# Link OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(lenet5 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lenet5 PUBLIC ENABLE_OPENMP)
endif()

# Optimize build flags for performance
target_compile_options(lenet5 PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)

# Print configuration
if(USE_CUDA)
    message(STATUS "Building for Tesla P4 (Compute Capability 6.1)")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()