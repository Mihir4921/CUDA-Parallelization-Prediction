#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void matrixMulGPU(float *ad, float *bd, float *cd, int n, int TILE_WIDTH);
void matrixMulCPU(float *A, float *B, float *C, int N);

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s <matrix_size> <tile_width>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int TILE_WIDTH = atoi(argv[2]);
    unsigned long long int num_bytes = N * N * sizeof(float);
    double time_taken;
    clock_t start, end;

    float *h_A = (float *)malloc(num_bytes);
    float *h_B = (float *)malloc(num_bytes);
    float *h_C = (float *)malloc(num_bytes);
//    printf("before matrix init\n");
    // Initialize matrices A and B
    for (long long int i = 0; i < N * N; i++)
    {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
  //  printf("matrix initialzied\n");
/*
    printf("CPU sequential version:\n");
    start = clock();
    matrixMulCPU(h_A, h_B, h_C, N);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
*/
    printf("GPU version:\n");
    start = clock();
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, num_bytes);
    cudaMalloc(&d_B, num_bytes);
    cudaMalloc(&d_C, num_bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, num_bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
   
    //printf("here\n");
    // Launch the kernel with dynamic shared memory allocation
    long long int sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);  // Two matrices in shared memory
    matrixMulGPU<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, d_C, N, TILE_WIDTH);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, num_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);

    printf("Grid size (number of blocks): %lld\n", gridSize.x * gridSize.y);
    printf("Block size (threads per block): %lld\n", blockSize.x * blockSize.y);
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

void matrixMulCPU(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matrixMulGPU(float *ad, float *bd, float *cd, int N, int TILE_WIDTH)
{
    extern __shared__ float shared_data[];
    float* Ads = shared_data;
    float* Bds = &shared_data[TILE_WIDTH * TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    //printf("before loop in kernel\n");
    float Cvalue = 0;
    for (int m = 0; m < N / TILE_WIDTH; ++m)
    {
        Ads[ty * TILE_WIDTH + tx] = ad[Row * N + (m * TILE_WIDTH + tx)];
        Bds[ty * TILE_WIDTH + tx] = bd[(m * TILE_WIDTH + ty) * N + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Cvalue += Ads[ty * TILE_WIDTH + k] * Bds[k * TILE_WIDTH + tx];
        }
        __syncthreads();
       // printf("end of loop\n");
    }
    //printf("after loop\n");
    if (Row < N && Col < N) {
        cd[Row * N + Col] = Cvalue;
    }
}
