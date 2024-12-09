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
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int mask_width_half = mask_width / 2;
    if(const int col = blockIdx.x * blockDim.x + threadIdx.x; col < w && row < h){
        const int n_start_col = col - mask_width_half;
        const int n_start_row = row - mask_width_half;
        int pixVal = 0;

        for(int j = 0; j < mask_width; ++j){
            const int curr_row = n_start_row + j;
            for(int k = 0; k < mask_width; ++k){
                if(const int curr_col = n_start_col + k; curr_row > -1 && curr_row < h && curr_col > -1 && curr_col < w){
                    pixVal += in[curr_row * w + curr_col] * mask[j * mask_width + k];
                }
            }
        }
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
 * Print a matrix.
 * @param matrix Matrix to print.
 * @param height Height of the matrix.
 * @param width Width of the matrix.
 */
void print_matrix(const int* matrix, const int height, const int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * height + j]);
        }
        printf("\n");
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
 * Create a matrix with random values between 0 and random_upper_bound.
 * @param result Result matrix.
 * @param rows Number of rows of the output mask, the matrix should be square.
 * @param random_upper_bound Upper limit for random number generation, 10 by default.
 * @throw invalid_argument If mask is null.
 */
void create_random_matrix(int *result, const int rows, const int random_upper_bound = 10) {
    if (result == nullptr)
        throw invalid_argument("Result matrix cannot be null");
    // init bound
    const int boundary = rows * rows;
    // insert values
    for (int i = 0; i < boundary; ++i)
        result[i] = static_cast<int>(random()) % random_upper_bound;
}

/**
 * Create a constant matrix with a specific value.
 * @param result Constant matrix result.
 * @param rows Rows of the final matrix, it should be square.
 * @param value Value to fill the matrix.
 * @throw invalid_argument If result is null.
 */
void create_constant_matrix(int *result, const int rows, const int value) {
    if (result == nullptr)
        throw invalid_argument("Result matrix cannot be null");
    // init bound
    const int boundary = rows * rows;
    // insert values
    for (int i = 0; i < boundary; ++i)
        result[i] = value;
}

int main(int argc, char const *argv[]) {
    // init
    constexpr int matrix_boundary = MATRIX_SIZE * MATRIX_SIZE;
    int mask_width = 0, block_size;
    int *mask, *in, *out;
    float naive_gpu_elapsed_time_ms = 0.0;

    // try to get env variable
    try {
        mask_width = stoi(getenv("MASK_SIZE"));
    }
    catch (...) {
        printf("Error reading MASK_SIZE env variable; it must be an integer");
        return -1;
    }

    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        // reserve size
        cudaMallocManaged(reinterpret_cast<void **>(&mask), sizeof(int) * mask_width * mask_width);
        cudaMallocManaged(reinterpret_cast<void **>(&in), sizeof(int) * matrix_boundary);
        cudaMallocManaged(reinterpret_cast<void **>(&out), sizeof(int) * matrix_boundary);

        // create mask matrix
        // create_mask_matrix(mask, mask_size);
        create_constant_matrix(mask, mask_width, 3);
        // create constant matrix (input)
        create_constant_matrix(in, MATRIX_SIZE, 2);
        // initialize output matrix
        create_constant_matrix(out, MATRIX_SIZE, 0);

        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        unsigned int grid_rows = (MATRIX_SIZE + block_size - 1) / block_size;
        unsigned int grid_cols = (MATRIX_SIZE + block_size - 1) / block_size;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(block_size, block_size);


        cudaEventRecord(start, nullptr);
        convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(in, mask, out, mask_width, MATRIX_SIZE, MATRIX_SIZE);
        cudaDeviceSynchronize();

        // time counting terminate

        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);

        // debug uncomment:
        // print_matrix(out, MATRIX_SIZE, MATRIX_SIZE);
        // print result
        printf("Time elapsed on naive GPU 2D-convolution of a matrix %dx%d using a mask %dx%d (block size %d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, mask_width, mask_width, block_size, naive_gpu_elapsed_time_ms);

        // free memory
        cudaFree(mask);
        cudaFree(in);
        cudaFree(out);
    }

    printf("\n\n ---------- TILING ---------- \n\n");
    // get dynamic tile
    const int tiling = get_dynamic_tile_width();
    // tiling already depends on the block size of the device,
    // so it is not necessary to include it in the for loop
    const int shared_memory_size = (tiling + mask_width - 1) * (tiling + mask_width - 1) * sizeof(int);

    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        // reserve size
        cudaMallocManaged(reinterpret_cast<void **>(&mask), sizeof(int) * mask_width * mask_width);
        cudaMallocManaged(reinterpret_cast<void **>(&in), sizeof(int) * matrix_boundary);
        cudaMallocManaged(reinterpret_cast<void **>(&out), sizeof(int) * matrix_boundary);

        // create mask matrix
        create_constant_matrix(mask, mask_width, 3);
        // create constant matrix (input)
        create_constant_matrix(in, MATRIX_SIZE, 2);
        // initialize output matrix
        create_constant_matrix(out, MATRIX_SIZE, 0);

        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, nullptr);

        dim3 dimBlock(tiling, tiling);
        dim3 dimGrid((MATRIX_SIZE + tiling - 1) / tiling, (MATRIX_SIZE + tiling - 1) / tiling);

        convolution_2D_tiled_kernel<<<dimGrid, dimBlock, shared_memory_size>>>(in, out, mask, mask_width, MATRIX_SIZE, MATRIX_SIZE, tiling);
        cudaDeviceSynchronize();

        // time counting terminate

        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);

        // debug uncomment:
        // print_matrix(out, MATRIX_SIZE, MATRIX_SIZE);
        // print result
        printf("Time elapsed on naive GPU 2D-convolution of a matrix %dx%d using a mask %dx%d (block size %d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, mask_width, mask_width, block_size, naive_gpu_elapsed_time_ms);

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