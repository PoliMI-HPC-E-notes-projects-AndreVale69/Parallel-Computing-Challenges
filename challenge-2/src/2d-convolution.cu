#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <fstream>

#define MATRIX_SIZE 16384

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

__global__ void convolution_2D_tiled_kernel(
    const int *in,
    int *out,
    const int *mask,
    const int mask_width,
    const int w,
    const int h,
    const int tile_width
) {
    extern __shared__ int N_ds[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_o = blockIdx.y * tile_width + ty;
    const int col_o = blockIdx.x * tile_width + tx;
    const int row_i = row_o - mask_width / 2;
    const int col_i = col_o - mask_width / 2;

    if ((row_i >= 0) && (row_i < h) && (col_i >= 0) && (col_i < w)) {
        N_ds[ty * tile_width + tx] = in[row_i * w + col_i];
    } else {
        N_ds[ty * tile_width + tx] = 0;
    }

    __syncthreads();

    if (ty < tile_width && tx < tile_width) {
        int output = 0;
        for (int i = 0; i < mask_width; ++i) {
            for (int j = 0; j < mask_width; ++j) {
                output += mask[i * mask_width + j] * N_ds[(i + ty) * tile_width + (j + tx)];
            }
        }
        if (row_o < h && col_o < w) {
            out[row_o * w + col_o] = output;
        }
    }
}

/**
 * Get the dynamic tile width for the current device.
 * @return Tile width.
 */
int get_dynamic_tile_width() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int tileWidth = static_cast<int>(sqrt(maxThreadsPerBlock));
    return tileWidth;
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

    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        // reserve size
        const int mask_boundary = mask_size * mask_size;
        const int out_boundary = mask_size + MATRIX_SIZE - 1;
        cudaMallocManaged(reinterpret_cast<void **>(&mask), sizeof(int) * mask_boundary * mask_boundary);
        cudaMallocManaged(reinterpret_cast<void **>(&in), sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
        cudaMallocManaged(reinterpret_cast<void **>(&out), sizeof(int) * out_boundary * out_boundary);

        // create mask matrix
        // create_mask_matrix(mask, mask_size);
        create_constant_matrix(mask, mask_size, 3);
        // create constant matrix (input)
        create_constant_matrix(in, MATRIX_SIZE, 2);
        // initialize output matrix
        create_constant_matrix(out, out_boundary, 0);


        float  naive_gpu_elapsed_time_ms;

        // some events to count the execution time
        //clock_t st, end;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        unsigned int grid_rows = (MATRIX_SIZE + block_size - 1) / block_size;
        unsigned int grid_cols = (MATRIX_SIZE + block_size - 1) / block_size;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(block_size, block_size);


        cudaEventRecord(start, nullptr);
        convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(in, mask, out, mask_size, MATRIX_SIZE, MATRIX_SIZE);
        cudaDeviceSynchronize();

        // time counting terminate

        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, block_size, naive_gpu_elapsed_time_ms);

        // free memory
        cudaFree(mask);
        cudaFree(in);
        cudaFree(out);
    }

    printf("\n\n ----- TILING: \n\n");
    const int tiling = get_dynamic_tile_width();
    const int shared_memory_size = tiling * tiling * sizeof(int);

    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        // reserve size
        const int mask_boundary = mask_size * mask_size;
        const int out_boundary = mask_size + MATRIX_SIZE - 1;
        cudaMallocManaged(reinterpret_cast<void **>(&mask), sizeof(int) * mask_boundary * mask_boundary);
        cudaMallocManaged(reinterpret_cast<void **>(&in), sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
        cudaMallocManaged(reinterpret_cast<void **>(&out), sizeof(int) * out_boundary * out_boundary);

        // create mask matrix
        // create_mask_matrix(mask, mask_size);
        create_constant_matrix(mask, mask_size, 3);
        // create constant matrix (input)
        create_constant_matrix(in, MATRIX_SIZE, 2);
        // initialize output matrix
        create_constant_matrix(out, out_boundary, 0);


        float  naive_gpu_elapsed_time_ms;

        // some events to count the execution time
        //clock_t st, end;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        unsigned int grid_rows = (MATRIX_SIZE + block_size - 1) / block_size;
        unsigned int grid_cols = (MATRIX_SIZE + block_size - 1) / block_size;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(block_size, block_size);


        cudaEventRecord(start, nullptr);

        convolution_2D_tiled_kernel<<<dimGrid, dimBlock, shared_memory_size>>>(in, out, mask, mask_size, MATRIX_SIZE, MATRIX_SIZE, tiling);
        cudaDeviceSynchronize();

        // time counting terminate

        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, block_size, naive_gpu_elapsed_time_ms);

        // free memory
        cudaFree(mask);
        cudaFree(in);
        cudaFree(out);
    }

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