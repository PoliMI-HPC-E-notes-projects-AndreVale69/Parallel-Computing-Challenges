#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <random>

#define MATRIX_SIZE 5
#define CPU_MATRIX_SIZE 1024

using namespace std;

__global__ void convolution_2D_basic_kernel(
    const int * in,
    const int * mask,
    int * out,
    const int mask_width,
    const int w,
    const int h
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (const int row = blockIdx.y * blockDim.y + threadIdx.y; col < w && row < h) {
        int pixVal = 0;

        const int n_start_col = col - (mask_width / 2);
        const int n_start_row = row - (mask_width / 2);

        // Get the sum of the surrounding box
        for(int j = 0, cur_row = n_start_row; j < mask_width; ++j, ++cur_row) {
            const int offset = cur_row * w;
            const int mask_offset = j * mask_width;
            for(int k = 0, cur_col = n_start_col; k < mask_width; ++k, ++cur_col) {
                // Verify we have a valid image pixel
                if(cur_row > -1 && cur_row < h && cur_col > -1 && cur_col < w) {
                    pixVal += in[offset + cur_col] * mask[mask_offset + k];
                }
            }
        }

        // Write our new pixel value out
        out[row * w + col] = pixVal;
    }
}

/**
 * Create a mask matrix with random values between 0 and random_upper_bound.
 * @param mask Vector mask, it will be modified locally.
 * @param rows Number of rows of the output mask, the matrix should be square.
 * @param random_upper_bound Upper limit for random number generation, 10 by default.
 * @throw invalid_argument If mask is null.
 */
void create_mask_matrix(int *mask, const int rows, const int random_upper_bound = 10) {
    if (mask == nullptr)
        throw invalid_argument("Mask matrix cannot be null");
    // init boud
    const int boundary = rows * rows;
    // generate random values from 0 to 10
    for(int i = 0; i < boundary; ++i)
        mask[i] = static_cast<int>(random()) % random_upper_bound;
}

/**
 * Create a constant matrix with a specific value.
 * @param result Constant matrix result.
 * @param rows Rows of the final matrix, it should be square.
 * @param value Value to fill the matrix.
 * @throw invalid_argument If result is null.
 */
int* create_constant_matrix(int *result, const int rows, const int value) {
    if (result == nullptr)
        throw invalid_argument("Result matrix cannot be null");
    // init bound
    const int boundary = rows * rows;
    // insert values
    for (int i = 0; i < boundary; ++i)
        result[i] = value;
    return result;
}

int main(int argc, char const *argv[]) {
    // init
    constexpr int matrix_boundary = MATRIX_SIZE * MATRIX_SIZE;
    int mask_size = 0, block_size;
    int *mask, *in, *out;

    // try to get env variable
    try {
        mask_size = stoi(getenv("MASK_SIZE"));
    }
    catch (...) {
        printf("Error reading MASK_SIZE env variable; it must be an integer");
        return -1;
    }
    // reserve size
    const int mask_boundary = mask_size * mask_size;
    const int out_boundary = mask_size + MATRIX_SIZE - 1;
    mask = static_cast<int *>(malloc(sizeof(int) * mask_boundary * mask_boundary));
    in = static_cast<int *>(malloc(sizeof(int) * matrix_boundary * matrix_boundary));
    out = static_cast<int *>(malloc(sizeof(int) * out_boundary * out_boundary));
    // create mask matrix
    create_mask_matrix(mask, mask_size);
    // create constant matrix (input)
    create_constant_matrix(in, MATRIX_SIZE, 1);
    // initialize output matrix
    create_constant_matrix(out, out_boundary, 0);

    // debug: print values
    for (int i = 0; i < mask_boundary; ++i)
        printf("mask[%d] = %d\n", i, mask[i]);

    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; ++i)
        printf("in[%d] = %d\n", i, in[i]);

    for (int i = 0; i < out_boundary; ++i)
        printf("out[%d] = %d\n", i, out[i]);

    // free
    free(mask);
    free(in);
    free(out);

    // retrieve some info about the CUDA device
    cudaGetDeviceCount(nullptr);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
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

    return 0;
}