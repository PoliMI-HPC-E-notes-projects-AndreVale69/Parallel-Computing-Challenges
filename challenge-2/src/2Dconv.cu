#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void convolution_2d(int* input, int* output, int height, int width, int* mask, int mask_dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int mask_radius = mask_dim / 2;

    if (row < height && col < width) {
        int result = 0;

        for (int i = -mask_radius; i <= mask_radius; i++) {
            for (int j = -mask_radius; j <= mask_radius; j++) {
                int cur_row = row + i;
                int cur_col = col + j;

                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
                    result += input[cur_row * width + cur_col] * mask[(i + mask_radius) * mask_dim + (j + mask_radius)];
                }
            }
        }

        output[row * width + col] = result;
    }
}


// Funzione per la convoluzione 2D sulla CPU
void convolution_2d_cpu(int *matrix, int *result, int height, int width, int *mask, int mask_dim) {
    int mask_offset = mask_dim / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int temp = 0;
            for (int k = 0; k < mask_dim; k++) {
                for (int l = 0; l < mask_dim; l++) {
                    int r = i - mask_offset + k;
                    int c = j - mask_offset + l;
                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        temp += matrix[r * width + c] * mask[k * mask_dim + l];
                    }
                }
            }
            result[i * width + j] = temp;
        }
    }
}

void initialize_matrix(int* matrix, int height, int width, int value) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i * width + j] = value;
        }
    }
}

void initialize_mask(int* mask, int mask_dim, int value) {
    for (int i = 0; i < mask_dim; i++) {
        for (int j = 0; j < mask_dim; j++) {
            mask[i * mask_dim + j] = value;
        }
    }
}


// Funzione per verificare il risultato
void verify_result(int *matrix, int *mask, int *result, int height, int width, int mask_dim) {
    int mask_offset = mask_dim / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int temp = 0;
            for (int k = 0; k < mask_dim; k++) {
                for (int l = 0; l < mask_dim; l++) {
                    int r = i - mask_offset + k;
                    int c = j - mask_offset + l;
                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        temp += matrix[r * width + c] * mask[k * mask_dim + l];
                    }
                }
            }
            assert(result[i * width + j] == temp);
        }
    }
}

// print a matrix
void print_matrix(const int *matrix, int rows, int cols, const std::string &label) {
    std::cout << label << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_height> <matrix_width> <mask_dim>\n";
        return 1;
    }

    int height = std::atoi(argv[1]);
    int width = std::atoi(argv[2]);
    int mask_dim = std::atoi(argv[3]);

    if (height <= 0 || width <= 0 || mask_dim <= 0 || mask_dim % 2 == 0) {
        std::cerr << "Error: Dimensions must be positive and mask_dim must be odd.\n";
        return 1;
    }

    size_t matrix_size = height * width * sizeof(int);
    size_t mask_size = mask_dim * mask_dim * sizeof(int);

    int* h_input = new int[height * width];
    int* h_output = new int[height * width];
    int* h_mask = new int[mask_dim * mask_dim];

    initialize_matrix(h_input, height, width, 1);
    initialize_mask(h_mask, mask_dim, 1);

    int *d_input, *d_output, *d_mask;
    check_cuda_error(cudaMalloc(&d_input, matrix_size), "cudaMalloc d_input");
    check_cuda_error(cudaMalloc(&d_output, matrix_size), "cudaMalloc d_output");
    check_cuda_error(cudaMalloc(&d_mask, mask_size), "cudaMalloc d_mask");

    check_cuda_error(cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice), "cudaMemcpy d_input");
    check_cuda_error(cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice), "cudaMemcpy d_mask");

    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

      // No tiling version
    auto start_no_tiling = std::chrono::high_resolution_clock::now();
    convolution_2d<<<grid_dim, block_dim>>>(d_input, d_output, height, width, d_mask, mask_dim);
    check_cuda_error(cudaDeviceSynchronize(), "Kernel execution");
    auto end_no_tiling = std::chrono::high_resolution_clock::now();

    check_cuda_error(cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost), "cudaMemcpy h_output");
    verify_result(h_input, h_mask, h_output, height, width, mask_dim);
     std::cout << "VERIFY COMPLETED SUCCESSFULLY!\n";
    //std::cout << "Output Matrix:\n";
    //for (int i = 0; i < height; i++) {
       // for (int j = 0; j < width; j++) {
         //   std::cout << h_output[i * width + j] << " ";
       // }
        //std::cout << "\n";
    //}
    //print_matrix(h_output, height, width, "result on d2conv on GPU");

      // CPU version
    int* result_cpu = new int[height * width];
    auto start_cpu = std::chrono::high_resolution_clock::now();
    convolution_2d_cpu(h_input, result_cpu, height, width, h_mask, mask_dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
   // print_matrix(result_cpu, height, width, "result on 2dconv on CPU");



 std::cout << "Time (GPU No Tiling): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_no_tiling - start_no_tiling).count()
              << " ms\n";

    std::cout << "Time (CPU): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count()
              << " ms\n";


    delete[] h_input;
    delete[] h_output;
    delete[] h_mask;
    check_cuda_error(cudaFree(d_input), "cudaFree d_input");
    check_cuda_error(cudaFree(d_output), "cudaFree d_output");
    check_cuda_error(cudaFree(d_mask), "cudaFree d_mask");

    return 0;
}