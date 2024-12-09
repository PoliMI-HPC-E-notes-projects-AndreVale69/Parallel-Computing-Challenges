#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define MATRIX_SIZE 8192
#define CPU_MATRIX_SIZE 1024


__global__ void gpu_matrix_mult(int *a,int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < n)
    {
        int sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void cpu_matrix_mult (int *a, int *b, int *c, int n)
{
    int i,j,k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            int sum_mult = 0;
            for (k = 0; k < n; k++)
            {
                sum_mult += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = sum_mult;
        }
    }
}

int main(int argc, char const *argv[])
{
    int block_size;

    /// retrieve some info about the CUDA device
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
    {
        int *a, *b, *c;
        a = (int*)malloc(sizeof(int)*CPU_MATRIX_SIZE*CPU_MATRIX_SIZE);
        b = (int*)malloc(sizeof(int)*CPU_MATRIX_SIZE*CPU_MATRIX_SIZE);
        c = (int*)malloc(sizeof(int)*CPU_MATRIX_SIZE*CPU_MATRIX_SIZE);
        // initialize matrix A
        for (int i = 0; i < CPU_MATRIX_SIZE; ++i) {
            for (int j = 0; j < CPU_MATRIX_SIZE; ++j) {
                a[i * CPU_MATRIX_SIZE + j] = 2;
            }
        }
        // initialize matrix B
        for (int i = 0; i < CPU_MATRIX_SIZE; ++i) {
            for (int j = 0; j < CPU_MATRIX_SIZE; ++j) {
                b[i * CPU_MATRIX_SIZE + j] = 3;
            }
        }
        // sequential version of matrix multiplication
        clock_t begin = clock();
        cpu_matrix_mult(a, b, c, CPU_MATRIX_SIZE);
        clock_t end = clock();
        double time_spent = ((double)((end - begin)) * 1000) / CLOCKS_PER_SEC;
        printf("Time elapsed on naive CPU sequential matrix multiplication of %dx%d . %dx%d: %f ms\n\n", CPU_MATRIX_SIZE, CPU_MATRIX_SIZE, CPU_MATRIX_SIZE, CPU_MATRIX_SIZE, time_spent);
        free(a);
        free(b);
        free(c);
    }

    for(block_size= 4; block_size <= 32; block_size *= 2)
    {
        int *a, *b, *c;
        cudaMallocManaged((void **) &a, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
        cudaMallocManaged((void **) &b, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
        cudaMallocManaged((void **) &c, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);

        // initialize matrix A
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                a[i * MATRIX_SIZE + j] = 2;
            }
        }

        // initialize matrix B
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                b[i * MATRIX_SIZE + j] = 3;
            }
        }


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


        cudaEventRecord(start, 0);
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, MATRIX_SIZE);
        cudaThreadSynchronize();

        // time counting terminate

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, block_size, naive_gpu_elapsed_time_ms);


        // free memory
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }

    return 0;
}