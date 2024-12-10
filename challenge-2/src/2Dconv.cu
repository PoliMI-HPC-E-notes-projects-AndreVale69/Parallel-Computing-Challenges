#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

// Kernel per la convoluzione 2D con tiling
__global__ void convolution_2d_tiled(int *matrix, int *result, int height, int width, int *mask, int mask_dim) {
    int mask_offset = mask_dim / 2;

    // Calcola posizione globale del thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Dimensione della memoria condivisa (tile + bordi)
    extern __shared__ int shared_matrix[];
    int shared_dim_x = blockDim.x + mask_dim - 1;
    int shared_dim_y = blockDim.y + mask_dim - 1;

    // Posizioni locali
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int shared_row = local_row + mask_offset;
    int shared_col = local_col + mask_offset;

    // Copia i dati in memoria condivisa con gestione dei bordi
    if (row < height && col < width) {
        shared_matrix[shared_row * shared_dim_x + shared_col] = matrix[row * width + col];
    } else {
        shared_matrix[shared_row * shared_dim_x + shared_col] = 0;
    }

    if (local_row < mask_offset && row >= mask_offset) {
        shared_matrix[(shared_row - mask_offset) * shared_dim_x + shared_col] = matrix[(row - mask_offset) * width + col];
    }
    if (local_row >= blockDim.y - mask_offset && row < height - mask_offset) {
        shared_matrix[(shared_row + mask_offset) * shared_dim_x + shared_col] = matrix[(row + mask_offset) * width + col];
    }

    if (local_col < mask_offset && col >= mask_offset) {
        shared_matrix[shared_row * shared_dim_x + (shared_col - mask_offset)] = matrix[row * width + (col - mask_offset)];
    }
    if (local_col >= blockDim.x - mask_offset && col < width - mask_offset) {
        shared_matrix[shared_row * shared_dim_x + (shared_col + mask_offset)] = matrix[row * width + (col + mask_offset)];
    }

    __syncthreads();

    // Esegui la convoluzione
    int temp = 0;
    if (row < height && col < width) {
        for (int i = 0; i < mask_dim; i++) {
            for (int j = 0; j < mask_dim; j++) {
                temp += shared_matrix[(shared_row - mask_offset + i) * shared_dim_x +
                                      (shared_col - mask_offset + j)] *
                        mask[i * mask_dim + j];
            }
        }
        result[row * width + col] = temp;
    }
}

// Funzione per inizializzare una matrice
void init_matrix(int *m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i * cols + j] = rand() % 10;
        }
    }
}

// Funzione per inizializzare una matrice (constante)
void init_matrix_constant(int *m, int rows, int cols, int constant) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i * cols + j] = constant;
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

// Funzione per scegliere dinamicamente blocchi e griglie
//versione precedente
/*void choose_optimal_tile_and_block(int &block_size, int &shared_memory_size, int mask_dim) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Massimo numero di thread per blocco
    int max_threads_per_block = prop.maxThreadsPerBlock;

    // Calcola dimensioni ottimali (quadrato vicino alla radice di max_threads_per_block)
    int optimal_block_size = sqrt(max_threads_per_block);
    block_size = optimal_block_size;

    // Dimensione della memoria condivisa (in base alla dimensione della maschera)
    shared_memory_size = (block_size + mask_dim - 1) * (block_size + mask_dim - 1) * sizeof(int);
}*/

void choose_optimal_tile_and_block(int &block_size, int &shared_memory_size, int mask_dim) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Massimo numero di thread per blocco e per SM
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

    // Numero massimo di blocchi per SM
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;

    // Memoria condivisa massima per blocco
    int max_shared_memory = prop.sharedMemPerBlock;

    // Calcola dimensioni ottimali (quadrato vicino alla radice di max_threads_per_block)
    int optimal_block_size = sqrt(max_threads_per_block);

    // Stima iniziale della memoria condivisa richiesta
    int required_shared_memory = (optimal_block_size + mask_dim - 1) * (optimal_block_size + mask_dim - 1) * sizeof(int);

    // Riduci il blocco se la memoria condivisa richiesta eccede quella disponibile
    while (required_shared_memory > max_shared_memory && optimal_block_size > 1) {
        optimal_block_size--;
        required_shared_memory = (optimal_block_size + mask_dim - 1) * (optimal_block_size + mask_dim - 1) * sizeof(int);
    }

    // Calcola il numero massimo di thread per blocco in base ai limiti di SM
    int max_threads_per_tile = max_threads_per_sm / max_blocks_per_sm;

    // Riduci il blocco se i thread per blocco eccedono i limiti del SM
    while ((optimal_block_size * optimal_block_size) > max_threads_per_tile && optimal_block_size > 1) {
        optimal_block_size--;
    }

    // Assegna le dimensioni finali
    block_size = optimal_block_size;
    shared_memory_size = (block_size + mask_dim - 1) * (block_size + mask_dim - 1) * sizeof(int);

    // Stampa informazioni utili per il debug
    std::cout << "Optimal block size: " << block_size << "x" << block_size << std::endl;
    std::cout << "Shared memory per block: " << shared_memory_size << " bytes" << std::endl;
    std::cout << "Threads per block: " << block_size * block_size << std::endl;
    std::cout << "Max threads per SM: " << max_threads_per_sm << std::endl;
    std::cout << "Max blocks per SM: " << max_blocks_per_sm << std::endl;
}


// Funzione per stampare una matrice
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

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_height> <matrix_width> <mask_dim>\n";
        return 1;
    }

    int height = std::atoi(argv[1]);     // Altezza della matrice
    int width = std::atoi(argv[2]);      // Larghezza della matrice
    int mask_dim = std::atoi(argv[3]);   // Dimensione della maschera

    size_t matrix_size = height * width * sizeof(int);
    size_t mask_size = mask_dim * mask_dim * sizeof(int);

    int *matrix = new int[height * width];
    int *mask = new int[mask_dim * mask_dim];
    int *result = new int[height * width];

    //init_matrix(matrix, height, width);
    init_matrix_constant(matrix, width, height, 1);
    //init_matrix(mask, mask_dim, mask_dim);
    init_matrix_constant(mask, mask_dim, mask_dim, 1);

    int *d_matrix, *d_result, *d_mask;
    cudaMalloc(&d_matrix, matrix_size);
    cudaMalloc(&d_result, matrix_size);
    cudaMalloc(&d_mask, mask_size);

    cudaMemcpy(d_matrix, matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, mask_size, cudaMemcpyHostToDevice);

    int block_size, shared_memory_size;
    choose_optimal_tile_and_block(block_size, shared_memory_size, mask_dim);

    int grid_size_x = (width + block_size - 1) / block_size;
    int grid_size_y = (height + block_size - 1) / block_size;
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim(grid_size_x, grid_size_y);

    convolution_2d_tiled<<<grid_dim, block_dim, shared_memory_size>>>(d_matrix, d_result, height, width, d_mask, mask_dim);

    cudaMemcpy(result, d_result, matrix_size, cudaMemcpyDeviceToHost);

    // Stampa la matrice iniziale, la maschera e il risultato
    print_matrix(matrix, height, width, "Input Matrix");
    print_matrix(mask, mask_dim, mask_dim, "Mask");
    print_matrix(result, height, width, "Result Matrix");

    verify_result(matrix, mask, result, height, width, mask_dim);

    std::cout << "Convolution completed successfully!" << std::endl;

    delete[] matrix;
    delete[] mask;
    delete[] result;
    cudaFree(d_matrix);
    cudaFree(d_result);
    cudaFree(d_mask);

    return 0;
}
