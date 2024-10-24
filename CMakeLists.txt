cmake_minimum_required(VERSION 3.29)
project(Parallel_Computing_Challenges)

set(CMAKE_CXX_STANDARD 20)
add_executable(sequential_merge_sort challenge-1/src/sequential_merge_sort.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")
add_executable(parallel_merge_sort challenge-1/src/parallel_merge_sort.cpp)
add_executable(alt_parallel_merge_sort challenge-1/src/alternative-solution-parallel_merge_sort.cpp)
