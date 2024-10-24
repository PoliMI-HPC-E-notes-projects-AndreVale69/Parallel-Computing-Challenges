cmake_minimum_required(VERSION 3.29)
project(Parallel_Computing_Challenges)

set(CMAKE_CXX_STANDARD 20)
add_executable(sequential_mergesort challenge-1/src/sequential_mergesort.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")
add_executable(parallel_mergesort challenge-1/src/mergesort.cpp)
