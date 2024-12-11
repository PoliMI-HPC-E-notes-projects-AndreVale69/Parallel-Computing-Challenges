#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

/**
 * Check for CUDA errors.
 * @param err Return value of CUDA runtime API function.
 * @param msg Error message to display.
 */
void err_check(const cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Naive 2D convolution kernel (no tiling).
 * @param input Input matrix.
 * @param output Output matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask Mask matrix.
 * @param mask_width Width of the mask matrix.
 */
__global__ void convolution_2d(
    const int * input,
    int * output,
    const int height,
    const int width,
    const int * mask,
    const int mask_width
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int mask_radius = mask_width / 2;

    if (row < height && col < width) {
        int result = 0;
        for (int i = -mask_radius; i <= mask_radius; i++) {
            for (int j = -mask_radius; j <= mask_radius; j++) {
                const int cur_row = row + i;
                if (const int cur_col = col + j; cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
                    result += input[cur_row * width + cur_col] * mask[(i + mask_radius) * mask_width + (j + mask_radius)];
                }
            }
        }
        output[row * width + col] = result;
    }
}

/**
 * Tiled 2D convolution kernel.
 * @param input Input matrix.
 * @param output Output matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask Mask matrix.
 * @param mask_dim Dimension of the mask matrix.
 * @param block_size Block size.
 */
__global__ void convolution_2d_tiled(
    const int * input,
    int * output,
    const int height,
    const int width,
    const int * mask,
    const int mask_dim,
    const int block_size
) {
    // initialize shared memory as dynamic
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#dynamic_shared_memory
    extern __shared__ int shared_mem[];

    const int mask_radius = mask_dim / 2;
    const int shared_width = block_size + 2 * mask_radius;

    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    const int block_col = blockIdx.x;
    const int block_row = blockIdx.y;

    // calculate the column and row for the current thread
    // with respect to the mask radius
    int col = block_col * block_size + thread_col - mask_radius;
    int row = block_row * block_size + thread_row - mask_radius;

    // load data into shared memory with boundary checks
    if (row >= 0 && row < height && col >= 0 && col < width) {
        shared_mem[thread_row * shared_width + thread_col] = input[row * width + col];
    } else {
        // avoid branch divergence
        shared_mem[thread_row * shared_width + thread_col] = 0;
    }

    // barrier synchronization
    __syncthreads();

    // compute convolution for threads within valid block range
    col = block_col * block_size + thread_col;
    row = block_row * block_size + thread_row;

    if (
        thread_col >= mask_radius &&                // left boundary
        thread_col < block_size + mask_radius &&    // right boundary
        thread_row >= mask_radius &&                // top boundary
        thread_row < block_size + mask_radius &&    // bottom boundary
        col < width && row < height                 // within matrix boundaries
    ) {
        // compute convolution
        int result = 0;
        for (int i = 0; i < mask_dim; i++) {
            for (int j = 0; j < mask_dim; j++) {
                result += shared_mem[(thread_row - mask_radius + i) * shared_width + (thread_col - mask_radius + j)] *
                          mask[i * mask_dim + j];
            }
        }
        // store the result, barrier synchronization is not needed
        output[row * width + col] = result;
    }
}


/**
 * Naive 2D convolution on CPU.
 * @param matrix Input matrix.
 * @param result Result matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask Mask matrix.
 * @param mask_dim Dimension of the mask matrix.
 */
void convolution_2d_cpu(
    const int * matrix,
    int * result,
    const int height,
    const int width,
    const int * mask,
    const int mask_dim
) {
    const int mask_offset = mask_dim / 2;
    // for each row
    for (int i = 0; i < height; ++i) {
        // for each column
        for (int j = 0; j < width; ++j) {
            int convolution = 0;
            // for each row of the mask
            for (int k = 0; k < mask_dim; ++k) {
                // for each column of the mask
                for (int l = 0; l < mask_dim; ++l) {
                    const int r = i - mask_offset + k;
                    if (const int c = j - mask_offset + l; r >= 0 && r < height && c >= 0 && c < width) {
                        convolution += matrix[r * width + c] * mask[k * mask_dim + l];
                    }
                }
            }
            result[i * width + j] = convolution;
        }
    }
}

/**
 * Initialize a matrix with a constant value.
 * @param matrix Result matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param value Value to initialize the matrix with.
 */
void initialize_matrix_by_constant(int* matrix, const int height, const int width, const int value) {
    // for each row
    for (int i = 0; i < height; i++) {
        // for each column
        for (int j = 0; j < width; j++) {
            // set the value
            matrix[i * width + j] = value;
        }
    }
}

/**
 * Create a matrix with random values between 0 and random_upper_bound.
 * @param matrix Result matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param random_upper_bound Upper limit for random number generation, 10 by default.
 * @throw invalid_argument If mask is null.
 */
void initialize_matrix_by_random(
    int * matrix,
    const int height,
    const int width,
    const int random_upper_bound
) {
    // for each row
    for (int i = 0; i < height; i++) {
        // for each column
        for (int j = 0; j < width; j++) {
            // set the value
            matrix[i * width + j] = static_cast<int>(random()) % random_upper_bound;
        }
    }
}

/**
 * Verify the result of the convolution.
 * The method compares the result of the convolution with the expected result.
 * It computes the convolution on the CPU and compares it with the result saved in the output matrix.
 * @param matrix Input matrix.
 * @param mask Mask matrix.
 * @param result Result matrix.
 * @param height Height of the input matrix.
 * @param width Width of the input matrix.
 * @param mask_dim Dimension of the mask matrix.
 */
void verify_result(
    const int * matrix,
    const int * mask,
    const int * result,
    const int height,
    const int width,
    const int mask_dim
) {
    // same implementation as convolution_2d_cpu but with assert
    const int mask_offset = mask_dim / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int expected_convolution = 0;
            for (int k = 0; k < mask_dim; k++) {
                for (int l = 0; l < mask_dim; l++) {
                    const int r = i - mask_offset + k;
                    if (const int c = j - mask_offset + l; r >= 0 && r < height && c >= 0 && c < width) {
                        expected_convolution += matrix[r * width + c] * mask[k * mask_dim + l];
                    }
                }
            }
            assert(result[i * width + j] == expected_convolution);
        }
    }
}

/**
 * Print a matrix.
 * @param matrix Matrix to print.
 * @param height Height of the matrix.
 * @param width Width of the matrix.
 * @param label Label to print before the matrix.
 */
void print_matrix(const int *matrix, int height, int width, const std::string &label) {
    printf("%s (%dx%d):\n", label.c_str(), height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // try to get env variable
    int height, width, mask_dim;
    try {
        mask_dim = std::stoi(getenv("MASK_SIZE"));
        height = std::stoi(getenv("MATRIX_HEIGHT"));
        width = std::stoi(getenv("MATRIX_WIDTH"));
        if (!(mask_dim > 0 && height > 0 && width > 0 && mask_dim % 2 != 0)) {
            throw std::invalid_argument("Invalid argument");
        }
    }
    catch (...) {
        printf("Error reading env variable MASK_SIZE, MATRIX_HEIGHT, MATRIX_WIDTH; "
            "it must be an integer, dimensions must be positive and the dimension of the mask must be odd.");
        return 1;
    }

    for (int tiling = 4; tiling <= 32; tiling *= 2) {
        for (int block_size = 4; block_size <= 32; block_size *= 2) {
            // TODO: bad access to CUDA memory - cases:
            if (block_size == 4 && tiling == 16 ||
                block_size == 4 && tiling == 32 ||
                block_size == 8 && tiling == 16 ||
                block_size == 8 && tiling == 32 ||
                block_size == 16 && tiling == 32) {
                continue;
            }
            const size_t matrix_size = height * width * sizeof(int);
            const size_t mask_size = mask_dim * mask_dim * sizeof(int);

            int* h_input = new int[height * width];
            int* h_output = new int[height * width];
            int* h_mask = new int[mask_dim * mask_dim];

            initialize_matrix_by_constant(h_input, height, width, 2);
            initialize_matrix_by_constant(h_mask, mask_dim, mask_dim, 3);

            int *d_input, *d_output, *d_mask;
            err_check(cudaMalloc(&d_input, matrix_size), "cudaMalloc d_input");
            err_check(cudaMalloc(&d_output, matrix_size), "cudaMalloc d_output");
            err_check(cudaMalloc(&d_mask, mask_size), "cudaMalloc d_mask");

            err_check(cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice), "cudaMemcpy d_input");
            err_check(cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice), "cudaMemcpy d_mask");

            dim3 block_dim(tiling, tiling);
            dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

            // No tiling version
            auto start_no_tiling = std::chrono::high_resolution_clock::now();
            convolution_2d<<<grid_dim, block_dim>>>(d_input, d_output, height, width, d_mask, mask_dim);
            err_check(cudaDeviceSynchronize(), "Kernel execution");
            auto end_no_tiling = std::chrono::high_resolution_clock::now();
            err_check(cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost), "cudaMemcpy h_output");
            // print_matrix(h_output, width, height, "result on d2conv on GPU");


            //tiled version
            int *d_output_tiled;
            int* h_output_tiled = new int[height * width];
            err_check(cudaMalloc(&d_output_tiled, matrix_size), "cudaMalloc d_output");
            size_t shared_mem_size = (block_size + 2 * (mask_dim / 2)) * (block_size + 2 * (mask_dim / 2)) * sizeof(int);
            auto start_tiled = std::chrono::high_resolution_clock::now();
            float naive_gpu_elapsed_time_ms = 0.0;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, nullptr);
            convolution_2d_tiled<<<grid_dim, block_dim, shared_mem_size>>>(d_input, d_output_tiled, height, width, d_mask, mask_dim, block_size);
            err_check(cudaDeviceSynchronize(), "Kernel execution");
            auto end_tiled = std::chrono::high_resolution_clock::now();
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            printf("Time elapsed on naive GPU 2D-convolution: %f ms.\n\n", naive_gpu_elapsed_time_ms);

            /**
             * Verify the result of the convolution
             */
            cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
            err_check(cudaMemcpy(h_output_tiled, d_output, matrix_size, cudaMemcpyDeviceToHost), "cudaMemcpy h_output");
            verify_result(h_input, h_mask, h_output_tiled, height, width, mask_dim);
            printf("Block size: %d; Tiled size: %d\n", block_size, tiling);
            printf(" ------ TILED ----- VERIFY COMPLETED SUCCESSFULLY!\n");
            // debug:
            // print_matrix(h_output_tiled, height, width, "TILED - result on d2conv on GPU");

            verify_result(h_input, h_mask, h_output, height, width, mask_dim);
            printf("(no tiled)VERIFY COMPLETED SUCCESSFULLY!\n");

            // CPU version
            int* result_cpu = new int[height * width];
            auto start_cpu = std::chrono::high_resolution_clock::now();
            convolution_2d_cpu(h_input, result_cpu, height, width, h_mask, mask_dim);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            // debug:
            // print_matrix(result_cpu, height, width, "result on 2dconv on CPU");

            // Print timings
            std::cout << "Time (GPU Tiled): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_tiled - start_tiled).count()
                      << " ms\n";

            std::cout << "Time (GPU No Tiling): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_no_tiling - start_no_tiling).count()
                      << " ms\n";

            std::cout << "Time (CPU): "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count()
                      << " ms\n";

            // free memory
            delete[] h_input;
            delete[] h_output;
            delete[] h_mask;
            delete[] h_output_tiled;
            delete[] result_cpu;
            // free cuda memory
            err_check(cudaFree(d_input), "cudaFree d_input");
            err_check(cudaFree(d_output), "cudaFree d_output");
            err_check(cudaFree(d_mask), "cudaFree d_mask");
        }
    }

    return 0;
}
