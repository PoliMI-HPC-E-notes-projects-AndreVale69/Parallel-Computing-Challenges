cmake_minimum_required(VERSION 3.30)
project(challenge_2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(challenge_2 src/matrix-multiplication.cu)

set_target_properties(challenge_2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
