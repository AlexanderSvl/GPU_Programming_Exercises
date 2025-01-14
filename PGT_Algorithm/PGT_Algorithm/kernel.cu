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
//      =========================================
//      |   -100 to -75   |     Dark blue       |     -- Represents deep sea area
//      |   -75 to -50    |     Light blue      |     -- Represents shallow sea area
//      |   -50 to -25    |     Light yellow    |     -- Represents low sand area
//      |   -25 to 0      |     Dark yellow     |     -- Represents high sand area
//      |   0 to +25      |     Light green     |     -- Represents low land area
//      |   +25 to +50    |     Dark green      |     -- Represents high land area
//      |   +50 to +75    |     Light brown     |     -- Represents low mountain area
//      |   +75 to +100   |     Dark brown      |     -- Represents high mountain area
//      =========================================
//
// -- 3. Design an algorithm to integrate the terrain generation
// -- 4. We need a way to store the previous terrain index, so we can create a gradual terrain generation.

__constant__ float terrain_step = 1.0; // Step by which we are going to modify the terrain.

__device__ float4 float_to_color(float x)
{
    // Red, green, blue, opacity
    float4 color;

    if (x >= -100 && x < -75) // Dark blue
    {
        color = make_float4(0.0f, 6.0f, 115.0f, 0.8f);
    }
    else if (x >= -75 && x < -50) // Light blue
    {
        color = make_float4(0.0f, 83.0f, 255.0f, 0.8f);
    }
    else if (x >= -50 && x < -25) // Light yellow
    {
        color = make_float4(254.0f, 255.0f, 124.0f, 0.8f);
    }
    else if (x >= -25 && x < 0) // Dark yellow
    {
        color = make_float4(231.0f, 197.0f, 15.0f, 0.8f);
    }
    else if (x >= 0 && x < 25) // Light green
    {
        color = make_float4(30.0f, 255.0f, 17.0f, 0.8f);
    }
    else if (x >= 25 && x < 50) // Dark green
    {
        color = make_float4(6.0f, 104.0f, 0.0f, 0.8f);
    }
    else if (x >= 50 && x < 75) // Light brown
    {
        color = make_float4(196.0f, 164.0f, 132.0f, 0.8f);
    }
    else if (x >= 75 && x <= 100) // Dark brown
    {
        color = make_float4(105.0f, 66.0f, 28.0f, 0.8f);
    }
    else if (x > 100)// Purple, represent error with the index in the color generation
    {
        //color = make_float4(191.0f, 0.0f, 137.0f, 0.8f);
        color = make_float4(0.0f, 0.0f, 0.0f, 0.8f);
    }

    return color;
}

__global__ void generate_terrain(const int* height_map, float4* color_buffer, int width, int height, int render_width, int render_height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= render_width || y >= render_height) return;

    unsigned int index = y * render_width + x;

    // Scale the input height map to the lower resolution
    unsigned int src_x = static_cast<unsigned int>(x * (float)width / render_width);
    unsigned int src_y = static_cast<unsigned int>(y * (float)height / render_height);

    // Ensure src_x and src_y are within bounds
    if (src_x >= width || src_y >= height) return;

    unsigned int src_index = src_y * width + src_x;

    // Get height value from the scaled-down height map
    float height_value = static_cast<float>(height_map[src_index]);

    // Convert height to color
    color_buffer[index] = float_to_color(height_value);
}

// Device pointers
int* device_height_map = nullptr;
float4* device_color_buffer = nullptr;

extern "C"
{
    // Initialize device memory for height map and color buffer
    void init_device_height_map(int width, int height)
    {
        cudaMalloc(&device_height_map, width * height * sizeof(int));
        cudaMalloc(&device_color_buffer, width * height * sizeof(float4));
    }

    // Free device memory
    void free_device_height_map()
    {
        if (device_height_map) cudaFree(device_height_map);
        if (device_color_buffer) cudaFree(device_color_buffer);
    }

    // Copy host height map to device
    void copy_height_map_to_device(const int* host_height_map, int width, int height)
    {
        cudaMemcpy(device_height_map, host_height_map, width * height * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Launch the terrain generation kernel
    void launch_kernel(int width, int height, int render_width, int render_height)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((render_width + blockDim.x - 1) / blockDim.x, (render_height + blockDim.y - 1) / blockDim.y);

        generate_terrain << <gridDim, blockDim >> > (device_height_map, device_color_buffer, width, height, render_width, render_height);
        cudaDeviceSynchronize();
    }

    // Copy the color buffer back to the host
    void copy_color_buffer_to_host(Uint32* host_buffer, int width, int height)
    {
        float4* host_color_buffer = new float4[width * height];
        cudaMemcpy(host_color_buffer, device_color_buffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

        for (int i = 0; i < width * height; ++i)
        {
            float4 color = host_color_buffer[i];

            // Convert float4 RGBA to Uint32 ARGB
            Uint32 a = static_cast<Uint32>(color.w * 255.0f) << 24;
            Uint32 r = static_cast<Uint32>(color.x) << 16;
            Uint32 g = static_cast<Uint32>(color.y) << 8;
            Uint32 b = static_cast<Uint32>(color.z);
            host_buffer[i] = a | r | g | b;
        }

        delete[] host_color_buffer;
    }
}