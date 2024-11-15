cmake_minimum_required(VERSION 3.29)
project(lenet5)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
add_executable(lenet5 main.cpp dataloader.h dataloader.cpp)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(lenet5 PUBLIC -DENABLE_OPENMP)  # Define a flag if OpenMP is enabled
endif()