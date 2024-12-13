###############
# Challenge 1 #
###############
cmake_minimum_required(VERSION 3.30)
project(challenge-1)

set(CMAKE_CXX_STANDARD 20)
add_executable(sequential-mergesort challenge-1/src/sequential_mergesort.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")
add_executable(openmp-parallel-mergesort challenge-1/src/mergesort.cpp)

###############
# Challenge 2 #
###############
project(challenge-2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda-matrix-multiplication challenge-2/src/matrix-multiplication.cu)
add_executable(cuda-2d-convolution challenge-2/src/2d-convolution-basic.cu)
add_executable(cuda-2d-convolution-tiling challenge-2/src/2d-convolution-tiling.cu)

set_target_properties(
        cuda-2d-convolution-tiling cuda-matrix-multiplication cuda-2d-convolution PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
