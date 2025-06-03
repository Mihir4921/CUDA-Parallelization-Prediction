#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include <cstdlib>

using namespace std;

// CPU Convolution
void convolutionCPU(const vector<vector<float>>& image, const vector<vector<float>>& kernel, vector<float>& output, int width) {
    int kernelSize = kernel.size();
    int offset = kernelSize / 2;

    for (int i = offset; i < width - offset; i++) {
        for (int j = offset; j < width - offset; j++) {
            float sum = 0.0;
            for (int k = -offset; k <= offset; k++) {
                for (int l = -offset; l <= offset; l++) {
                    sum += image[i + k][j + l] * kernel[offset + k][offset + l];
                }
            }
            output[i * width + j] = sum;
        }
    }
}

// CUDA Kernel
__global__ void convolutionCUDA(float* d_image, float* d_kernel, float* d_output, int width, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = kernelSize / 2;

    float sum = 0.0f;
    if (x >= offset && x < width - offset && y >= offset && y < width - offset) {
        for (int i = -offset; i <= offset; i++) {
            for (int j = -offset; j <= offset; j++) {
                sum += d_image[(y + i) * width + (x + j)] * d_kernel[(i + offset) * kernelSize + (j + offset)];
            }
        }
        d_output[y * width + x] = sum;
    }
}

// GPU Convolution
void convolutionGPU(const vector<vector<float>>& image, const vector<vector<float>>& kernel, vector<float>& output, int width, int threads, int blocks) {
    int kernelSize = kernel.size();

    float *d_image, *d_kernel, *d_output;
    cudaMalloc(&d_image, width * width * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, width * width * sizeof(float));

    // Flatten image
    vector<float> flat_image(width * width);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            flat_image[i * width + j] = image[i][j];
        }
    }

    cudaMemcpy(d_image, flat_image.data(), width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel[0].data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(threads, threads);
    dim3 blocksPerGrid(blocks, blocks);


    convolutionCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_kernel, d_output, width, kernelSize);
    cudaDeviceSynchronize(); // Ensure the kernel execution is finished


    cudaMemcpy(output.data(), d_output, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <image_size> <kernel_size> <num_threads> <num_blocks> <0 for CPU, 1 for GPU>" << endl;
        return 1;
    }

    int imageSize = atoi(argv[1]);
    int kernelSize = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int blocks = atoi(argv[4]);
    int useGPU = atoi(argv[5]);

    if (kernelSize % 2 == 0) {
        cerr << "Kernel size must be an odd number." << endl;
        return 1;
    }

    vector<vector<float>> image(imageSize, vector<float>(imageSize, 1.0));
    vector<vector<float>> kernel(kernelSize, vector<float>(kernelSize, 1.0 / (kernelSize * kernelSize)));
    vector<float> output(imageSize * imageSize, 0.0);

    if (useGPU == 0) {
        clock_t start = clock();
        convolutionCPU(image, kernel, output, imageSize);
        clock_t end = clock();
        double cpu_time = double(end - start) / CLOCKS_PER_SEC * 1000;
        cout << "CPU Time: " << cpu_time << " ms" << endl;
    } else {
        clock_t start = clock();
        convolutionGPU(image, kernel, output, imageSize, threads, blocks);
        clock_t end = clock();
        double gpu_time = double(end - start) / CLOCKS_PER_SEC * 1000;
        cout << "GPU Time: " << gpu_time << " ms" << endl;
    }

    return 0;
}
