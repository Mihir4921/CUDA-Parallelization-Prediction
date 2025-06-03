#include <cuda.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#define DEBUG 0

// kernel for DFT
/*
__global__ void DFTKernel(const cuComplex* input, cuComplex* output, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        cuComplex sum = make_cuComplex(0.0, 0.0);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * M_PI * k * n / N;
            cuComplex exp_term = make_cuComplex(cosf(angle), sinf(angle));
            sum = cuCaddf(sum, cuCmulf(input[n], exp_term));
        }
        output[k] = sum;
    }
}
*/

__global__ void DFTKernel(const cuComplex* input, cuComplex* output, int N) {
    int totalThreads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int k = tid; k < N; k += totalThreads) {
        cuComplex sum = make_cuComplex(0.0, 0.0);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * M_PI * k * n / N;
            cuComplex exp_term = make_cuComplex(cosf(angle), sinf(angle));
            sum = cuCaddf(sum, cuCmulf(input[n], exp_term));
        }
        output[k] = sum;
    }
}

// CPU ver
void DFTSequential(const cuComplex* input, cuComplex* output, int N) {
    for (int k = 0; k < N; ++k) {
        cuComplex sum = make_cuComplex(0.0, 0.0);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * M_PI * k * n / N;
            cuComplex exp_term = make_cuComplex(cos(angle), sin(angle));
            sum = cuCaddf(sum, cuCmulf(input[n], exp_term));
        }
        output[k] = sum;
    }
}


void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <problem_size> <device> <num_blocks> <threadsPerBlock>\n", argv[0]);
        return -1;
    }

    int problemSize = atoi(argv[1]);
    int device = atoi(argv[2]);
    int numBlocks = atoi(argv[3]);
    int numThreadsPerBlock = atoi(argv[4]);
    
    int verbose = 0;

    if (numThreadsPerBlock <= 0 || numBlocks <= 0 || problemSize <= 0) {
        printf("All arguments must be positive integers.\n");
        return -1;
    }

    
    cuComplex *h_input, *h_output_seq, *h_output_gpu;
    h_input = (cuComplex*)malloc(problemSize * sizeof(cuComplex));
    h_output_seq = (cuComplex*)malloc(problemSize * sizeof(cuComplex));
    h_output_gpu = (cuComplex*)malloc(problemSize * sizeof(cuComplex));

    
    for (int i = 0; i < problemSize; i++) {
        h_input[i] = make_cuComplex((float)i, 0.0f);
    }

    if(!device){
    
	    clock_t start_cpu = clock();
	    DFTSequential(h_input, h_output_seq, problemSize);
	    clock_t end_cpu = clock();
	    double elapsed_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
	    printf("CPU Sequential DFT Time: %f seconds\n", elapsed_cpu);
    }

    cuComplex *d_input, *d_output;
    cudaMalloc(&d_input, problemSize * sizeof(cuComplex));
    cudaMalloc(&d_output, problemSize * sizeof(cuComplex));

    checkCudaError("Memory allocation failed");


    clock_t start_gpu = clock();
    cudaMemcpy(d_input, h_input, problemSize * sizeof(cuComplex), cudaMemcpyHostToDevice);

#ifdef DEBUG
    checkCudaError("Memory copy to device failed");
#endif
    
    DFTKernel<<<numBlocks, numThreadsPerBlock>>>(d_input, d_output, problemSize);

#ifdef DEBUG
    printf("debug\n");
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");
#endif

    //printf("HERE\n");
    cudaMemcpy(h_output_gpu, d_output, problemSize * sizeof(cuComplex), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    checkCudaError("Memory copy to host failed");
#endif

    clock_t end_gpu = clock();
    double elapsed_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    if(verbose){
    	printf("GPU Version:\n");
    }
    printf("time taken: %f\n", elapsed_gpu);


    free(h_input);
    free(h_output_seq);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
