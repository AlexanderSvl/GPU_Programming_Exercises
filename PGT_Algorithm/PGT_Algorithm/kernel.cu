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

__constant__ float terrain_step = 1.0; // Step by which we are going to modify the terrain.

float4 float_to_color(float x)
{
    // Red, green, blue, opacity
    float4 color;

    if (x >= -10 && x < -5) // Dark blue
    {
        color = make_float4(0.0f, 6.0f, 115.0f, 0.8f);
    }
    else if (x >= -5 && x < 0) // Light blue
    {
        color = make_float4(0.0f, 83.0f, 255.0f, 0.8f);
    }
    else if (x >= 0 && x < 5) // Light yellow
    {
        color = make_float4(254.0f, 255.0f, 124.0f, 0.8f);
    }
    else if (x >= 5 && x < 10) // Dark yellow
    {
        color = make_float4(231.0f, 197.0f, 15.0f, 0.8f);
    }
    else if (x >= 10 && x < 15) // Light green
    {
        color = make_float4(30.0f, 255.0f, 17.0f, 0.8f);
    }
    else if (x >= 15 && x < 20) // Dark green
    {
        color = make_float4(6.0f, 104.0f, 0.0f, 0.8f);
    }
    else if (x >= 20 && x < 25) // Light brown
    {
        color = make_float4(196.0f, 164.0f, 132.0f, 0.8f);
    }
    else if (x >= 25 && x < 30) // Dark brown
    {
        color = make_float4(105.0f, 66.0f, 28.0f, 0.8f);
    }
    else // Purple, represent error with the index in the color generation
    {
        color = make_float4(191.0f, 0.0f, 137.0f, 0.8f);
    }

    return color;
}

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

        generate_terrain <<<gridDim, blockDim >>> (terrain, width, height);
        cudaDeviceSynchronize();
    }
}