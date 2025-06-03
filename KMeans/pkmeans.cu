#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <float.h>


void shuffle(float* points, size_t n);
float* randArr(int n);
float* initCentroids(float* points, int k);
float distance(float* p1, float* p2);
int* assignCluster(float* points, float* centroids, int k);
float* updateCentroids(float* points, int* clusters);
int compare(float* arr1, float* arr2);
float* KMeans(float* points, int k, int maxIters);
float* PKM(float* points, int k, int maxIters, float tol);

__global__ void getClusters(float*, float*, float*, int*, int*, int, int, int);
__global__ void updateC(float*, float*, int*, float*, int, int);

size_t numPoints, k;
int DIMS = 2;
int numBlocks, tpBlock;
int verbose = 0;

int main(int argc, char* argv[]){


	if(argc != 6 && argc != 4){
		printf("Usage: ./kmeans <n> <k> <device>:[<numBlocks> <threadsPerBlock>]\n");
		printf("	n = number of points\n");
		printf("	k = number of centroids\n");
		printf("	device: CPU=0, GPU=1\n");
		printf("	numBlocks = number of blocks\n");
		printf("	threadsPerBlock <= 1024\n");
		exit(1);
	}

	float *points, *centroids;
	float tol = 0.00001;	
	int maxIters = 1000;
	clock_t start, end;

	
	numPoints = atoi(argv[1]);
	k = atoi(argv[2]);
	int device = atoi(argv[3]); //0 for CPU, 1 for GPU
	if(device){
		numBlocks = atoi(argv[4]);
		tpBlock = atoi(argv[5]);
	}

	if(verbose){
		printf("KMeans Clustering on %d elements\n", numPoints);
		printf("--------------------------------------\n");
	}
	
	points = randArr(numPoints);
	
	/*
	for(int i = 0; i < numPoints; i++){
		printf("points[%d]: (", i);
		for(int j = 0; j < DIMS; j++){
			printf("%f, ", points[i * DIMS + j]);
		}
		printf(")\n");
	}
	*/

	if(device == 0){
		if(verbose)
			printf("CPU Version\n");

		start = clock();
		centroids = KMeans(points, k, maxIters);
		end = clock();
	}else{
		if(verbose)
			printf("GPU Version\n");

		start = clock();
		centroids = PKM(points, k, maxIters, tol);
		end = clock();
	}

	double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
	printf("Execution time: %f\n", time_taken);	

	if(verbose){
	//	printf("Time taken = %lf\n", time_taken);
		printf("--------------------------------------\n");
	}

	/*
	for(int i = 0; i < k; i++){
		printf("centroids[%d]: (", i);
		for(int j = 0; j < DIMS; j++){
			printf("%f, ", centroids[i * DIMS + j]);
		}
		printf(")\n");
	}
	printf("--------------------------------------\n");
	*/
}


void shuffle(float* points, size_t n) {
   for (size_t i = n - 1; i > 0; i--) {
       size_t j = rand() % (i + 1);

       for (int d = 0; d < DIMS; d++) {
           float temp = points[i * DIMS + d];
           points[i * DIMS + d] = points[j * DIMS + d];
           points[j * DIMS + d] = temp;
       }
   }
}


float* randArr(int n) {
   float* pts = (float*)malloc(n * DIMS * sizeof(float));
   srand((unsigned int)time(NULL));

   if (!pts) {
       printf("error allocating array pts\n");
       exit(1);
   }

   for(int i = 0; i < n; i++) {
       for(int j = 0; j < DIMS; j++) {
           pts[i * DIMS + j] = ((float)rand() / RAND_MAX * 200.0f) - 100.0f;
//	   pts[i * DIMS + j] = rand() % 11;
       }
   }

   return pts;
}


float* initCentroids(float* points, int k) {

   float* centroids = (float*)malloc(k * DIMS * sizeof(float));

   if (!centroids) {
       printf("error allocating array centroids\n");
       exit(1);
   }

   shuffle(points, numPoints);

   for(int i = 0; i < k; i++) {
       for(int d = 0; d < DIMS; d++) {
           centroids[i * DIMS + d] = points[i * DIMS + d];
       }
   }

   return centroids;
}


__host__ __device__ float distance(float* points, float* centroids, int pidx, int cidx, int DIMS){
	
	float dist = 0;
	for(int i = 0; i < DIMS; i++){
		float diff = points[pidx * DIMS + i] - centroids[cidx * DIMS + i];
		dist += diff * diff;
	}

	return sqrt(dist);
}

int* assignCluster(float* points, float* centroids, int k){
	
	int* clusters = (int*)malloc(numPoints * sizeof(int));
	
	for(int i = 0; i < numPoints; i++){
		float minDist = FLT_MAX;
		int ctr = -1;

		for(int j = 0; j < k; j++){

			float dist = distance(points, centroids, i, j, DIMS); 

			if(dist < minDist){
				minDist = dist;
				ctr = j;
			}

	
		}
	
		clusters[i] = ctr;
	}

	return clusters;
}


float* updateCentroids(float* points, int* clusters) {
    float* newCentroids = (float*)calloc(k * DIMS, sizeof(float));
    int* counts = (int*)calloc(k, sizeof(int));

    for (int j = 0; j < numPoints; j++) {
        int clusterId = clusters[j];
        counts[clusterId]++;

        for (int d = 0; d < DIMS; d++) {
            newCentroids[clusterId * DIMS + d] += points[j * DIMS + d];
        }
    }

    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (int d = 0; d < DIMS; d++) {
                newCentroids[i * DIMS + d] /= counts[i];
            }
        }
    }

    free(counts);
    return newCentroids;
}


int compare(float* arr1, float* arr2){

	for(int i = 0; i < k; i++){
		for(int j = 0; j < DIMS; j++){
			if(arr1[i * DIMS + j] != arr2[i * DIMS + j]){
				return 0;
			}
		}
	}

	return 1;	

}

/*
float* KMeans(float* points, int k, int maxIters){
	
	int* clusters = (int*)malloc(numPoints * sizeof(int));
	float* centroids = initCentroids(points, k);
	int converged = 0, count = 0;

	while(!converged && count < maxIters){
		clusters = assignCluster(points, centroids, k);
		float* centroids2 = updateCentroids(points, clusters);
		
		if(compare(centroids, centroids2))
			converged = 1;
	
		free(centroids);
		centroids = centroids2;
		
		count++;

	}
	printf("iters taken: %d\n", count);

	return centroids;
} 
*/


float* KMeans(float* points, int k, int maxIters){
    int* clusters = (int*)malloc(numPoints * sizeof(int));
    int* counts = (int*)calloc(k, sizeof(int));
    float* centroids = initCentroids(points, k);
    float* centroids2 = (float*)calloc(k * DIMS, sizeof(float));
    int count = 0;
    float maxDelta;

    do {
        memset(counts, 0, k * sizeof(int));
        memset(centroids2, 0, k * DIMS * sizeof(float));
        maxDelta = 0.0f;

        for(int i = 0; i < numPoints; i++){
            float minDist = FLT_MAX;
            int minC = -1;

            for(int j = 0; j < k; j++){
                float dist = distance(points, centroids, i, j, DIMS);
                if(dist < minDist){
                    minDist = dist;
                    minC = j;
                }
            }
            clusters[i] = minC;
            counts[minC]++;
            for(int d = 0; d < DIMS; d++) {
                centroids2[minC * DIMS + d] += points[i * DIMS + d];
            }
        }

        // Update centroids and check for convergence
        for(int i = 0; i < k; i++) {
            if(counts[i] > 0) {  // Only update if cluster has points
                float maxChange = 0.0f;
                for(int d = 0; d < DIMS; d++) {
                    float oldValue = centroids[i * DIMS + d];
                    centroids2[i * DIMS + d] /= counts[i];
                    float change = fabs(centroids2[i * DIMS + d] - oldValue);
                    maxChange = fmax(maxChange, change);
                    centroids[i * DIMS + d] = centroids2[i * DIMS + d];
                }
                maxDelta = fmax(maxDelta, maxChange);
            }
        }

        count++;
    } while(count < maxIters && maxDelta > 0.00001);
//    } while(maxDelta > 0.0001);
	
    if(verbose)
	printf("iters taken: %d\n", count);

    free(clusters);
    free(counts);
    free(centroids2);

    return centroids;
}

__global__ void getClusters(float *points_d, float *centroids_d, float *centroids2_d, int* clusters_d, int* counts_d, int n, int k, int DIMS){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int s = blockDim.x * gridDim.x;


	for(int i = idx; i < n; i += s){
		float minDist = FLT_MAX;
		int minC = -1;

		for(int j = 0; j < k; j++){
			float dist = distance(points_d, centroids_d, i, j, DIMS); 
		//	printf("points_d[%d]: (%f, %f)\n", i, points_d[i * DIMS + 0], points_d[i * DIMS + 1]);
		//	printf("cenroids_d[%d]: (%f, %f)\n", i, centroids_d[i * DIMS + 0], centroids_d[i * DIMS + 1]);
		//	printf("DIST: %f\n", dist);
			if(dist < minDist){
				minDist = dist;
				minC = j;
			}	
		}

		clusters_d[i] = minC;
		atomicAdd(&counts_d[minC], 1);
//		printf("Cluster[%d]: %d\n", i, clusters_d[i]);
		for(int d = 0; d < DIMS; d++) {
         	   atomicAdd(&centroids2_d[minC * DIMS + d], points_d[i * DIMS + d]);
        	}
	}	
}


__global__ void updateC(float* centroids_d, float* newCentroids_d, int* counts_d, float* maxDelta_d, int k, int DIMS) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= k) return;
    
    if (counts_d[idx] > 0) {
        float maxChange = 0.0f;
        float count = static_cast<float>(counts_d[idx]);
        
        for (int d = 0; d < DIMS; d++) {
            float oldValue = centroids_d[idx * DIMS + d];
            float newValue = newCentroids_d[idx * DIMS + d] / count;
            float change = fabs(newValue - oldValue);
            maxChange = max(maxChange, change);
            centroids_d[idx * DIMS + d] = newValue;
        }
        
        atomicMax((unsigned int*)maxDelta_d, __float_as_int(maxChange));
    }
}


float* PKM(float* points, int k, int maxIters, float tol){
	
	//wont need later
	int *clusters = (int*)malloc(numPoints * sizeof(int));
	int *counts = (int*)malloc(k * sizeof(int));
	float* centroids2 = (float*)malloc(k * DIMS * sizeof(float));
	float* centroids = initCentroids(points, k);

	float *points_d, *centroids_d, *centroids2_d, *maxDelta_d;
	int *clusters_d, *counts_d;
	float maxDelta;


	cudaMalloc((void**)&points_d, numPoints * DIMS * sizeof(float));
	cudaMalloc((void**)&centroids_d, k * DIMS * sizeof(float));
	cudaMalloc((void**)&centroids2_d, k * DIMS * sizeof(float));
	cudaMalloc((void**)&clusters_d, numPoints * sizeof(int));
	cudaMalloc((void**)&counts_d, k * sizeof(int));
	cudaMalloc((void**)&maxDelta_d, sizeof(float));

	cudaMemcpy(points_d, points, numPoints * DIMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids, k * DIMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(counts_d, counts, k * DIMS * sizeof(int), cudaMemcpyHostToDevice);

	int count = 0;

	do{
		cudaMemset(centroids2_d, 0, k * DIMS * sizeof(float));
		cudaMemset(counts_d, 0, k * sizeof(int));
		cudaMemset(maxDelta_d, 0, sizeof(float));

		getClusters<<<numBlocks, tpBlock>>>(points_d, centroids_d, centroids2_d, clusters_d, counts_d, numPoints, k, DIMS);
		cudaDeviceSynchronize();

		updateC<<<numBlocks, tpBlock>>>(centroids_d, centroids2_d, counts_d, maxDelta_d, k, DIMS);
		cudaDeviceSynchronize();


		cudaMemcpy(&maxDelta, maxDelta_d, sizeof(float), cudaMemcpyDeviceToHost);

		count++;
	
		//if(count % 100 == 0)
		//	printf("iter: %d\n", count);
//	}while(count < maxIters && maxDelta > tol);
    	} while(maxDelta > 0.001);

	printf("iters taken: %d\n", count);

	cudaMemcpy(centroids, centroids_d, k * DIMS * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(points_d);
	cudaFree(centroids_d);
	cudaFree(centroids2_d);
	cudaFree(clusters_d);
	cudaFree(counts_d);
	cudaFree(maxDelta_d);

	// Free host memory
	free(clusters);
	free(counts);
	free(centroids2);

	return centroids;	
}



