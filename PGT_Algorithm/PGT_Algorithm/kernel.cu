#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// Steps to integrate the PGT algorithm
// 
// -- 1. Determine the range in which the terrain generates (e.g -10 to +20, -10 being underwater, +20 being a mountain)
// -- 2. Determine the colors in the different ranges of the terrain.
// 
//                    COLOR TABLE
//      =======================================
//      |   -10 to -5   |     Dark blue       |     -- Represents deep sea area
//      |    -5 to 0    |     Light blue      |     -- Represents shallow sea area
//      |   0 to +5     |     Light yellow    |     -- Represents low sand area
//      |   +5 to +10   |     Dark yellow     |     -- Represents high sand area
//      |   +10 to +15  |     Light green     |     -- Represents low land area
//      |   +15 to +20  |     Dark green      |     -- Represents high land area
//      |   +20 to +25  |     Light brown     |     -- Represents low mountain area
//      |   +25 to +30  |     Dark brown      |     -- Represents high mountain area
//      =======================================
//
// -- 3. Design an algorithm to integrate the terrain generation
// -- 4. We need a way to store the previous terrain index, so we can create a gradual terrain generation.

__global__ void generate_terrain(float* terrain, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Implement the PTG Algorithm
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