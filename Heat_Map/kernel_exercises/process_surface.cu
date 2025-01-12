#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void processSurfaceKernel(float* heatmap, float* newHeatmap, int width, int height, float dissipationRate, int mouseX, int mouseY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        // Heat dissipation calculation
        float currentHeat = heatmap[index];
        float dissipatedHeat = currentHeat * dissipationRate;

        // Calculate average heat from neighbors (simple diffusion)
        float neighborHeat = 0.0f;
        int count = 0;

        if (x > 0)
        {
            neighborHeat += heatmap[index - 1];
            count++;
        }
        if (x < width - 1)
        {
            neighborHeat += heatmap[index + 1];
            count++;
        }
        if (y > 0)
        {
            neighborHeat += heatmap[index - width];
            count++;
        }
        if (y < height - 1)
        {
            neighborHeat += heatmap[index + width];
            count++;
        }

        neighborHeat /= max(1, count);

        // Combine dissipated heat with neighbor influence
        newHeatmap[index] = dissipatedHeat + (1.0f - dissipationRate) * neighborHeat;

        // Add heat to a larger area around the mouse position
        int radius = 50; // Increase this value to make the heat effect bigger
        int dx = x - mouseX;
        int dy = y - mouseY;

        if (dx * dx + dy * dy <= radius * radius) // Check if within circular area
        {
            newHeatmap[index] += 10.0f;
        }
    }
}

// Global device pointers for the heatmaps
float* devHeatmap = nullptr;
float* devNewHeatmap = nullptr;

extern "C"
{
    void initHeatmaps(int width, int height)
    {
        cudaMalloc(&devHeatmap, width * height * sizeof(float));
        cudaMalloc(&devNewHeatmap, width * height * sizeof(float));
        cudaMemset(devHeatmap, 0, width * height * sizeof(float));
        cudaMemset(devNewHeatmap, 0, width * height * sizeof(float));
    }

    void freeHeatmaps()
    {
        if (devHeatmap) cudaFree(devHeatmap);
        if (devNewHeatmap) cudaFree(devNewHeatmap);
    }

    void processSurface(int width, int height, int mouseX, int mouseY)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        processSurfaceKernel << <gridDim, blockDim >> > (devHeatmap, devNewHeatmap, width, height, 0.99f, mouseX, mouseY);
        cudaDeviceSynchronize();

        // Swap heatmap pointers
        float* temp = devHeatmap;
        devHeatmap = devNewHeatmap;
        devNewHeatmap = temp;
    }

    void copyHeatmapToHost(float* hostHeatmap, int width, int height)
    {
        cudaMemcpy(hostHeatmap, devHeatmap, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    }
}