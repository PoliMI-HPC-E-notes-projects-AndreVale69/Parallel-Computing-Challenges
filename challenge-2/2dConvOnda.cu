#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define CPU_MATRIX_SIZE 1024
#define MASK_WIDTH 3
#define MASK_HEIGHT 3
#define HEIGHT 1024
#define WIDTH 1024

__global__ void gpu_convolution(int *matrix,int *mask, int *res, int maskwidth, int w, int h)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < w && row< h){
        int pixVal = 0;
        int N_start_col = col - (maskwidth/2);
        int N_start_row = row - (maskwidth/2);

        for(int j = 0; j<maskwidth; ++j){
            for(int k = 0; k<maskwidth; ++k){
                int currRow = N_start_row +j;
                int currCol = N_start_col + k;

                if(currRow>-1 && currRow < h && currCol > -1 && currCol < w){
                    pixVal+= matrix[currRow * w + currCol] * mask[j*maskwidth+ k];
                }
            }
        }

        res[row*w+ col] = pixVal;
    }

    
}

void printMatrix(const int* matrix, int width, int height, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char const *argv[])
{
    int block_size;
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  max Blocks Per MultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
      printf("  max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
      printf("  max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
      printf("  num SM: %d\n", prop.multiProcessorCount);
      printf("  num bytes sharedMem Per Block: %d\n", prop.sharedMemPerBlock);
      printf("  num bytes sharedMem Per Multiprocessor: %d\n", prop.sharedMemPerMultiprocessor);
      printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        int *a, *b, *c;
        cudaMallocManaged((void **) &a, sizeof(int)*WIDTH*HEIGHT);
        cudaMallocManaged((void **) &b, sizeof(int)*MASK_WIDTH * MASK_HEIGHT);
        cudaMallocManaged((void **) &c, sizeof(int)*WIDTH*HEIGHT);

        // initialize matrix A
        for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                a[i * WIDTH + j] = 2;
            }
        }

        // initialize matrix B
        for (int i = 0; i < MASK_HEIGHT; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                b[i * MASK_WIDTH + j] = 3;
            }
        }

        // Stampa le matrici dopo l inizializzazione
        printMatrix(a, WIDTH, HEIGHT, "A");
        printMatrix(b, MASK_WIDTH, MASK_HEIGHT, "B");


        float  naive_gpu_elapsed_time_ms;

        // some events to count the execution time
        //clock_t st, end;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        unsigned int grid_cols = (WIDTH + block_size - 1) / block_size;
        unsigned int grid_rows = (HEIGHT + block_size - 1) / block_size;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(block_size, block_size);


        cudaEventRecord(start, 0);
        gpu_convolution<<<dimGrid, dimBlock>>>(a, b, c, MASK_WIDTH, WIDTH, HEIGHT);
        cudaDeviceSynchronize();
        // Stampa la matrice risultato
        printMatrix(c, WIDTH, HEIGHT, "C"); 
        // time counting terminate

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", HEIGHT, WIDTH, HEIGHT, WIDTH, block_size, naive_gpu_elapsed_time_ms);


        // free memory
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }
    return 0;





}
