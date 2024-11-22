cmake_minimum_required(VERSION 3.16)
project(lenet5)

set(CMAKE_CXX_STANDARD 20)

# Option to enable GPU support
option(ENABLE_GPU "Enable GPU support using CUDA" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
    message(STATUS "Configuring for macOS")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# If GPU is enabled, find CUDA
if(ENABLE_GPU)
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DENABLE_GPU)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Add source files
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

# If GPU is enabled, add GPU-specific source files
if(ENABLE_GPU)
    list(APPEND SOURCES ) # Add your GPU implementation files
endif()

# Add the executable
add_executable(lenet5 ${SOURCES})

# Link OpenMP and CUDA libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(lenet5 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lenet5 PUBLIC -DENABLE_OPENMP) # Define a flag if OpenMP is enabled
endif()

if(ENABLE_GPU)
    target_link_libraries(lenet5 PUBLIC ${CUDA_LIBRARIES})
    target_compile_definitions(lenet5 PUBLIC -DENABLE_GPU)
endif()
