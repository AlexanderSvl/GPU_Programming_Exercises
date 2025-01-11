#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void processSurfaceKernel(Uint32* pixels, int width, int height, int mouseX, int mouseY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int dx = x - mouseX;
        int dy = y - mouseY;

        if (dx * dx + dy * dy < 5 * 5)
        {
            pixels[index] = 0xFF0000;
        }
    }
}

extern "C" 
{
    void processSurface(Uint32* devPixels, int width, int height, int mouseX, int mouseY)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        processSurfaceKernel <<<gridDim, blockDim >>> (devPixels, width, height, mouseX, mouseY);
        cudaDeviceSynchronize();
    }
}