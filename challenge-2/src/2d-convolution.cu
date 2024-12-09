#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <random>

#define MATRIX_SIZE 8192
#define CPU_MATRIX_SIZE 1024

__global__ void convolution_global_memory(
    const float *n, const float *m, float *o, const int width, const int mask_width
) {

	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	float p_value = 0;

	const int n_start_point = i-(mask_width/2);

	for(int j = 0; j < mask_width; ++j)
		if(n_start_point + j >= 0 && n_start_point + j < width)
			p_value+= n[n_start_point + j] * m[j];

	o[i]=p_value;
}

__global__ void gpu_matrix_mult(const int *a, const int *b, int *c, const int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(const int col = blockIdx.x * blockDim.x + threadIdx.x; col < n && row < n)
    {
        const int offset = row * n;
        int sum = 0;
        for(int i = 0; i < n; ++i)
            sum += a[offset + i] * b[i * n + col];
        c[offset + col] = sum;
    }
}

/**
 * Create a mask matrix with random values between 0 and random_upper_bound.
 * @param mask Vector mask, it will be modified locally.
 * @param rows Number of rows of the output mask, the matrix should be square.
 * @param random_upper_bound Upper limit for random number generation, 10 by default.
 */
void create_mask_matrix(std::vector<int> *mask, const int rows, const int random_upper_bound = 10) {
    // init boud
    const int boundary = rows * rows;

    // reserve size
    mask->reserve(boundary);

    // generate random values from 0 to 10
    for(int _ = 0; _ < boundary; ++_)
        mask->emplace_back(random() % random_upper_bound);
}

int main(int argc, char const *argv[]) {
    // init
    int mask_size = 0, block_size, n_devices;
    std::vector<int> mask;

    // try to get env variable
    try {
        mask_size = std::stoi(std::getenv("MASK_SIZE"));
    }
    catch (...) {
        printf("Error reading MASK_SIZE env variable; it must be an integer");
        return -1;
    }

    // create mask matrix
    create_mask_matrix(&mask, mask_size);

    // debug: print values
    for (int i = 0; i < mask_size*mask_size; ++i)
        printf("mask[%d] = %d\n", i, mask.at(i));

    // retrieve some info about the CUDA device
    cudaGetDeviceCount(&n_devices);
    for (int i = 0; i < n_devices; ++i) {
      cudaDeviceProp prop{};
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  max Blocks Per MultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
      printf("  max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
      printf("  max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
      printf("  num SM: %d\n", prop.multiProcessorCount);
      printf("  num bytes sharedMem Per Block: %lu\n", prop.sharedMemPerBlock);
      printf("  num bytes sharedMem Per Multiprocessor: %lu\n", prop.sharedMemPerMultiprocessor);
      printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    return 0;
}