cmake_minimum_required(VERSION 3.30)
project(challenge-2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(challenge-2 src/2d-convolution.cu)

set_target_properties(challenge-2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(matrix-multiplication src/matrix-multiplication.cu)

set_target_properties(matrix-multiplication PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
