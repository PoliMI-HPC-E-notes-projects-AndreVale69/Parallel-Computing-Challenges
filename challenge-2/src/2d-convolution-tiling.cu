#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <fstream>

#define BLOCK_WIDTH 32

using namespace std;

/**
 * 2D convolution kernel with tiling.
 * It optimizes the convolution operation by using shared memory to store the input matrix tile,
 * reducing the number of global memory accesses.
 *
 * The kernel first loads a tile of the input matrix into shared memory,
 * then performs the convolution operation within the tile.
 *
 * @param cuda_input_m Input matrix.
 * @param mask Mask matrix.
 * @param cuda_output_m Output matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask_width Width of the mask matrix.
 * @param N_TILE_WIDTH Tile width.
 */
__global__ void convolution_2D_tiled_kernel(
    const float* cuda_input_m,
    const float* __restrict__ mask,
    float* cuda_output_m,
    const size_t height,
    const size_t width,
    const size_t mask_width,
    const int N_TILE_WIDTH
) {
    // shared memory
    __shared__ float tile_shared_memory[BLOCK_WIDTH][BLOCK_WIDTH];

    // init
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n_row = blockIdx.y * N_TILE_WIDTH + ty;
    const int n_col = blockIdx.x * N_TILE_WIDTH + tx;
    const int m_row = n_row - mask_width / 2;
    const int m_col = n_col - mask_width / 2;

    // boundary condition
    if(m_row >= 0 && m_row < height && m_col >= 0 && m_col < width) {
        // load element from input matrix to shared memory in the respective tile position
        tile_shared_memory[ty][tx] = cuda_input_m[m_row * width + m_col];
    } else {
        // avoid branch divergence
        tile_shared_memory[ty][tx] = 0;
    }

    // barrier synchronization
    __syncthreads();

    // boundary condition to avoid out-of-bounds access, because we calculate only N_TILE_LENGTH elements
    if(ty < N_TILE_WIDTH && tx < N_TILE_WIDTH && n_row < height && n_col < width)
    {
        // convolution result
        float convolution_result = 0;
        // calculate convolution result
        for(int i = 0; i < mask_width; ++i) {
            for(int j = 0; j < mask_width; ++j) {
                convolution_result += mask[i * mask_width + j] * tile_shared_memory[ty + i][tx + j];
            }
        }
        // save convolution result to output matrix (barrier synchronization not needed)
        cuda_output_m[n_row * width + n_col] = convolution_result;
    }
}


/**
 * Print a matrix.
 * @param matrix Matrix to print.
 * @param height Height of the matrix.
 * @param width Width of the matrix.
 */
void print_matrix(const float* matrix, const int height, const int width) {
    if (matrix == nullptr) {
        throw invalid_argument("Matrix cannot be null");
    }
    for (int i = 0; i < height * width; ++i) {
        printf("%f ", matrix[i]);
        if (i % width == width - 1) {
            printf("\n");
        }
    }
}

/**
 * Create a matrix with random values between 0 and upper_bound.
 * @param result Result matrix.
 * @param rows Number of rows of the output matrix.
 * @param cols Number of cols of the output matrix.
 * @param lower_bound Lower limit for random number generation.
 * @param upper_bound Upper limit for random number generation.
 * @throw invalid_argument If mask is null.
 */
void create_random_matrix(float *result, const int rows, const int cols, const int lower_bound, const int upper_bound) {
    if (lower_bound > upper_bound) {
        throw invalid_argument("Lower bound cannot be greater than upper bound");
    }
    if (result == nullptr) {
        throw invalid_argument("Result matrix cannot be null");
    }
    // init bound
    const int boundary = rows * cols;
    // insert values
    for (int i = 0; i < boundary; ++i)
        result[i] = ((random() % upper_bound) + lower_bound);
}

/**
 * Create a constant matrix with a specific value.
 * @param result Constant matrix result.
 * @param rows Rows of the final matrix, it should be square.
 * @param cols Cols of the final matrix, it should be square.
 * @param value Value to fill the matrix.
 * @throw invalid_argument If result is null.
 */
void create_constant_matrix(float *result, const int rows, const int cols, const float value) {
    if (result == nullptr)
        throw invalid_argument("Result matrix cannot be null");
    // init bound
    const int boundary = rows * cols;
    // insert values
    for (int i = 0; i < boundary; ++i)
        result[i] = value;
}

/**
 * Verify the result of the convolution operation using the CPU.
 * @param matrix Input matrix.
 * @param mask Mask matrix.
 * @param result Result matrix to verify.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask_dim Dimension of the mask matrix.
 */
void verify_result(
    const float *matrix,
    const float *mask,
    const float *result,
    const int height,
    const int width,
    const int mask_dim
) {
    if (matrix == nullptr || mask == nullptr || result == nullptr) {
        throw invalid_argument("Matrix, mask, and result cannot be null");
    }
    const int mask_offset = mask_dim / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float expected_convolution = 0.0;
            for (int k = 0; k < mask_dim; k++) {
                for (int l = 0; l < mask_dim; l++) {
                    const int r = i - mask_offset + k;
                    if (const int c = j - mask_offset + l; r >= 0 && r < height && c >= 0 && c < width) {
                        expected_convolution += matrix[r * width + c] * mask[k * mask_dim + l];
                    }
                }
            }
            const float convolution = result[i * width + j];
            assert(convolution == expected_convolution);
        }
    }
}


int main() {
    // init
    int mask_width = 0, matrix_width = 0, matrix_height = 0;
    int seed = 0, warmup = 0;

    // try to get env variable about the matrix size and mask size
    try {
        mask_width = stoi(getenv("MASK_SIZE"));
        matrix_width = stoi(getenv("MATRIX_WIDTH"));
        matrix_height = stoi(getenv("MATRIX_HEIGHT"));
        if (!(mask_width > 0 && matrix_width > 0 && matrix_height > 0 && mask_width % 2 != 0)) {
            throw invalid_argument("Invalid argument");
        }
    } catch (...) {
        printf("Error reading MASK_SIZE env variable; it must be an integer.\n");
        return 1;
    }
    // try to get env variable about the seed
    try {
        seed = stoi(getenv("SEED"));
    } catch (...) {
        printf("WARNING: SEED env variable not found; random values will be generated.\n\n");
    }
    // try to get env variable about the warmup
    try {
        warmup = stoi(getenv("WARMUP"));
    } catch (...) {
        printf("WARNING: WARMUP env variable not set, a single run will be performed.\n\n");
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


    /** Execution **/
    // init
    float naive_gpu_elapsed_time_ms;
    const int N_TILE_WIDTH = BLOCK_WIDTH - (mask_width - 1);
    float* input_m = static_cast<float *>(malloc(matrix_height * matrix_width * sizeof(float)));
    float* mask = static_cast<float *>(malloc(mask_width * mask_width * sizeof(float)));
    float* output_m = static_cast<float *>(malloc(matrix_width * matrix_height * sizeof(float)));

    // populate
    if (seed != 0) {
        srand(seed);
    }
    create_random_matrix(input_m, matrix_height, matrix_width, 1, 100);
    create_random_matrix(mask, mask_width, mask_width, 1, 5);

    // =============================================== START CONVOLUTION ===============================================
    // time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // define grid and block dimensions
    dim3 dim_grid(
        ceil(matrix_width / static_cast<float>(N_TILE_WIDTH)),
        ceil(matrix_height / static_cast<float>(N_TILE_WIDTH))
    );
    dim3 dim_block(BLOCK_WIDTH, BLOCK_WIDTH);
    const size_t bytes_input_m = matrix_height * matrix_width * sizeof(int);
    const size_t bytes_mask = mask_width * mask_width * sizeof(int);

    // allocate memory in the device
    float* cuda_input_m;
    float* cuda_mask;
    float* cuda_output_m;
    cudaMalloc(reinterpret_cast<void **>(&cuda_input_m), bytes_input_m);
    cudaMalloc(reinterpret_cast<void **>(&cuda_mask), bytes_mask);
    cudaMalloc(reinterpret_cast<void **>(&cuda_output_m), bytes_input_m); // same bytes as input

    // initialize memory in the device
    cudaMemcpy(cuda_input_m, input_m, bytes_input_m, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mask, mask, bytes_mask, cudaMemcpyHostToDevice);

    // warmup
    for (int i = 0; i < warmup; ++i) {
        convolution_2D_tiled_kernel<<<dim_grid, dim_block>>>(
            cuda_input_m, cuda_mask, cuda_output_m, matrix_height, matrix_width, mask_width, N_TILE_WIDTH
        );
    }

    cudaEventRecord(start, nullptr);
    convolution_2D_tiled_kernel<<<dim_grid, dim_block>>>(
        cuda_input_m, cuda_mask, cuda_output_m, matrix_height, matrix_width, mask_width, N_TILE_WIDTH
    );
    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    cudaMemcpy(output_m, cuda_output_m, bytes_input_m, cudaMemcpyDeviceToHost);


    cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);

    printf("Check the result of the convolution operation using the CPU...\n");
    verify_result(input_m, mask, output_m, matrix_height, matrix_width, mask_width);
    printf("Verification passed!\n");

    // debug: print the output matrix
    // print_matrix(output_m, matrix_height, matrix_width);
    printf("Time elapsed on naive GPU 2D-convolution of %dx%d (block %d): %f ms.\n\n",
        matrix_height, matrix_width, BLOCK_WIDTH, naive_gpu_elapsed_time_ms);

    cudaFree(cuda_input_m);
    cudaFree(cuda_mask);
    cudaFree(cuda_output_m);

    // ================================================ END CONVOLUTION ================================================

    // free
    free(input_m);
    free(mask);
    free(output_m);
    return 0;
}
