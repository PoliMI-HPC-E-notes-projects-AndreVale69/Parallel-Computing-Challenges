#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstdlib>
#include <iostream>

#define MATRIX_SIZE 8192
#define CPU_MATRIX_SIZE 1024

__global__ void convolution_global_memory(float *n, float *m, float *o, int width, const int mask_width){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	float p_value = 0;

	const int n_start_point = i-(mask_width/2);

	for(int j = 0; j < mask_width; ++j)
		if(n_start_point + j >= 0 && n_start_point + j < width)
			p_value+= n[n_start_point + j] * m[j];

	o[i]=p_value;
}

__global__ void gpu_matrix_mult(const int *a, const int *b, int *c, const int n)
{
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

void cpu_matrix_mult (const int *a, const int *b, int *c, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int offset = i * n;
            int sum_mult = 0;
            for (int k = 0; k < n; ++k)
                sum_mult += a[offset + k] * b[k * n + j];
            c[offset + j] = sum_mult;
        }
}

int main(int argc, char const *argv[])
{
    int mask_size = 0, block_size, n_devices;
    // try to get env variable
    try {
        mask_size = std::stoi(std::getenv("MASK_SIZE"));
    }
    catch (...) {
        printf("Error reading MASK_SIZE env variable; it must be an integer");
        return -1;
    }

    // retrieve some info about the CUDA device
    cudaGetDeviceCount(&n_devices);
    for (int i = 0; i < n_devices; ++i) {
      cudaDeviceProp prop;
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