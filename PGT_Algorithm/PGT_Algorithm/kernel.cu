#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

__global__ void generate_terrain(float* terrain, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C"
{
    void launch_kernel(float* terrain, int width, int height)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        generate_terrain << <gridDim, blockDim >> > (terrain, width, height);
        cudaDeviceSynchronize();
    }
}