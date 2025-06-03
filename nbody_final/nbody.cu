#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

using namespace std;

#define TIME_STEP 0.01f // Time step for simulation
#define G 6.67430e-11f  // Gravitational constant

// Structure to hold properties of each body
struct Body
{
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;       // Mass
};

// GPU Kernel function to compute forces and update positions
__global__ void updateBodiesGPU(Body *bodies, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float distSq = dx * dx + dy * dy + dz * dz + 1e-10f;
                float dist = sqrt(distSq);

                float force = G * bodies[i].mass * bodies[j].mass / distSq;
                fx += force * dx / dist;
                fy += force * dy / dist;
                fz += force * dz / dist;
            }
        }

        bodies[i].vx += fx / bodies[i].mass * dt;
        bodies[i].vy += fy / bodies[i].mass * dt;
        bodies[i].vz += fz / bodies[i].mass * dt;

        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

// CPU version of the N-body simulation
void updateBodiesCPU(Body *bodies, int n, float dt)
{
    for (int i = 0; i < n; i++)
    {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float distSq = dx * dx + dy * dy + dz * dz + 1e-10f;
                float dist = sqrt(distSq);

                float force = G * bodies[i].mass * bodies[j].mass / distSq;
                fx += force * dx / dist;
                fy += force * dy / dist;
                fz += force * dz / dist;
            }
        }

        bodies[i].vx += fx / bodies[i].mass * dt;
        bodies[i].vy += fy / bodies[i].mass * dt;
        bodies[i].vz += fz / bodies[i].mass * dt;

        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

// Initialize bodies with random positions and masses
void initializeBodies(Body *bodies, int n)
{
    for (int i = 0; i < n; i++)
    {
        bodies[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f - 50.0f;
        bodies[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f - 50.0f;
        bodies[i].z = static_cast<float>(rand()) / RAND_MAX * 100.0f - 50.0f;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0f;
        bodies[i].mass = static_cast<float>(rand()) / RAND_MAX * 10.0f + 1.0f;
    }
}

bool compareBodies(const Body *a, const Body *b, int n, float epsilon = 1e-5)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(a[i].x - b[i].x) > epsilon || fabs(a[i].y - b[i].y) > epsilon || fabs(a[i].z - b[i].z) > epsilon)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    // Check for command-line argument

    // Parse the number of bodies from command line
    int numBodies = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numBlocks = atoi(argv[3]);
//int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
    clock_t start, end;
    double time_taken;

    // Allocate memory for bodies on host
    Body *h_bodies = new Body[numBodies];
    Body *cpu_bodies = new Body[numBodies]; // Copy of bodies for CPU computation
    initializeBodies(h_bodies, numBodies);
    copy(h_bodies, h_bodies + numBodies, cpu_bodies); // Copy for CPU computation
/*
    printf("CPU sequential version:\n");
    start = clock();
    // Run the CPU simulation
    for (int step = 0; step < 100; step++)
    {
        updateBodiesCPU(cpu_bodies, numBodies, TIME_STEP);
    }
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
*/
    // Allocate memory for bodies on device
    printf("GPU version:\n");
    start = clock();

    Body *d_bodies;
    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Run the GPU simulation
    for (int step = 0; step < 100; step++)
    {
        updateBodiesGPU<<<numBlocks, threadsPerBlock>>>(d_bodies, numBodies, TIME_STEP);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_bodies, d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
/*
    // Compare results
    if (compareBodies(h_bodies, cpu_bodies, numBodies))
    {
        printf("GPU and CPU results match!\n");
    }
    else
    {
        printf("GPU and CPU results do not match!\n");
    }
*/
    printf("Number of blocks: %d\n", numBlocks);
    printf("Threads per block: %d\n", threadsPerBlock);

    // Free memory
    delete[] h_bodies;
    delete[] cpu_bodies;
    cudaFree(d_bodies);

    return 0;
}
