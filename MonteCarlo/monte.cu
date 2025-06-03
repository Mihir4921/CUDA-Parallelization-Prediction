#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void piGPU(curandState *states, int iterations, int *total_hits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockDim.x * gridDim.x; 
    
//    curand_init(clock64(), tid, 0, &states[tid]);
    curand_init(1234, tid, 0, &states[tid]);
    int hits = 0;

    for(int i = tid; i < iterations; i += s) {
        
        float x = curand_uniform(&states[tid]);
        float y = curand_uniform(&states[tid]);
        
        if (x*x + y*y <= 1.0f) {
        	hits++;
		//atomicAdd(total_hits, 1);
	}
    }
	atomicAdd(total_hits, hits);
}

double piCPU(int iterations) {
    int hits = 0;
    srand(time(NULL));
    
    for (int i = 0; i < iterations; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        
        if (x*x + y*y <= 1.0) {
            hits++;
        }
    }
    
    return 4.0 * hits / iterations;
}

int main(int argc, char* argv[]) {

	
	if(argc != 5){
		printf("Usage: ./monte <numIters> <device> <blocks> <threadsPerBlock>\n");

		exit(1);
	}	

	int iterations = atoi(argv[1]);
	int device = atoi(argv[2]);
	int blockSize = atoi(argv[3]);
	int numBlocks = atoi(argv[4]);

	int verbose = 0;

	if(!device){
	
		clock_t cpu_start = clock();
		double pi_cpu = piCPU(iterations);
		clock_t cpu_end = clock();
		double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
		printf("CPU Results:\n");
		printf("Estimated Pi: %f\n", pi_cpu);
		printf("time taken: %f seconds\n\n", cpu_time);
	}else{
	
		int *d_hits;
		curandState *d_states;
		int h_hits = 0;

		cudaMalloc(&d_hits, sizeof(int));
		cudaMalloc(&d_states, iterations * sizeof(curandState));
		cudaMemset(d_hits, 0, sizeof(int));


		clock_t start = clock(); 
		piGPU<<<numBlocks, blockSize>>>(d_states, iterations, d_hits);

		cudaMemcpy(&h_hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost);

		clock_t end = clock();

		float time = ((float)(end - start)) / CLOCKS_PER_SEC;
		double pi_gpu = 4.0 * h_hits / iterations;
		if(verbose){
		
			printf("GPU Results:\n");
			printf("Estimated Pi: %f\n", pi_gpu);
		}
		printf("time taken: %f\n", time);

		cudaFree(d_hits);
		cudaFree(d_states);
	}


	return 0;
}
