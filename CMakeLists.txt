cmake_minimum_required(VERSION 3.29)
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
project(lenet5)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
# add_executable(lenet5 main.cpp dataloader.cpp)
add_executable(lenet5 main.cpp lenet5.cpp dataloader.cpp FCLayer.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lenet5 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lenet5 PUBLIC -DENABLE_OPENMP)  # Define a flag if OpenMP is enabled
endif()