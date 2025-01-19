#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

//                    COLOR TABLE
//      ================================
//      |   -2   |     Dark blue       |     -- Represents deep sea area
//      |   -1   |     Light blue      |     -- Represents shallow sea area
//      |   0    |     Yellow          |     -- Represents sand area
//      |   1    |     Light green     |     -- Represents low land area
//      |   2    |     Dark green      |     -- Represents high land area
//      |   3    |     Grey            |     -- Represents low mountain area
//      |   4    |     White           |     -- Represents high mountain area (peaks)
//      ================================

__device__ float4 float_to_color(float x)
{
    // Red, green, blue, opacity
    float4 color;

    if (x == -2) // Dark blue
    {
        color = make_float4(0.0f, 56.0f, 168.0f, 0.8f);
    }
    else if (x == -1) // Light blue
    {
        color = make_float4(0.0f, 83.0f, 255.0f, 0.8f);
    }
    else if (x == 0) // Yellow
    {
        color = make_float4(254.0f, 255.0f, 124.0f, 0.8f);
    }
    else if (x == 1) // Light green
    {
        color = make_float4(30.0f, 255.0f, 17.0f, 0.8f);
    }
    else if (x == 2) // Dark green
    {
        color = make_float4(25.0f, 205.0f, 15.0f, 0.8f);
    }
    else if (x == 3) // Grey
    {
        color = make_float4(142.0f, 142.0f, 142.0f, 0.8f);
    }
    else if (x == 4) // White
    {
        color = make_float4(255.0f, 255.0f, 255.0f, 0.8f);
    }
    else // Purple, represents out-of-bounds index in the color generation
    {
        color = make_float4(191.0f, 0.0f, 137.0f, 0.8f);
        //color = make_float4(0.0f, 0.0f, 0.0f, 0.8f);
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