#include "terrain.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>

int global_seed = std::rand() % 40000001;; // static_cast<int>(time(0));  // Initialize seed once

float noise(float x, float y)
{
    int n = static_cast<int>(x + y * 57 + global_seed);  // Use the global seed
    n = (n << 13) ^ n;
    return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

float smoothNoise(float x, float y)
{
    float corners = (noise(x - 1, y - 1) + noise(x + 1, y - 1) + noise(x - 1, y + 1) + noise(x + 1, y + 1)) / 16;
    float sides = (noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) + noise(x, y + 1)) / 8;
    float center = noise(x, y) / 4;
    return corners + sides + center;
}

float interpolatedNoise(float x, float y)
{
    int integerX = static_cast<int>(x);
    float fractionalX = x - integerX;

    int integerY = static_cast<int>(y);
    float fractionalY = y - integerY;

    float v1 = smoothNoise(integerX, integerY);
    float v2 = smoothNoise(integerX + 1, integerY);
    float v3 = smoothNoise(integerX, integerY + 1);
    float v4 = smoothNoise(integerX + 1, integerY + 1);

    float i1 = (1 - fractionalX) * v1 + fractionalX * v2;
    float i2 = (1 - fractionalX) * v3 + fractionalX * v4;

    return (1 - fractionalY) * i1 + fractionalY * i2;
}

float perlinNoise(float x, float y, int octaves, float persistence)
{
    float total = 0;
    float frequency = 1;
    float amplitude = 1;
    float maxValue = 0;

    for (int i = 0; i < octaves; ++i)
    {
        total += interpolatedNoise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2;
    }

    return total / maxValue;
}

// Helper function to enforce gradual transitions
void enforce_gradual_transitions(int** height_map, unsigned width, unsigned height)
{
    for (unsigned y = 0; y < height; ++y)
    {
        for (unsigned x = 0; x < width; ++x)
        {
            int current = height_map[y][x];

            // Check all neighbors and enforce the rule
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (nx >= 0 && nx < (int)width && ny >= 0 && ny < (int)height && !(dx == 0 && dy == 0))
                    {
                        int neighbor = height_map[ny][nx];
                        if (abs(neighbor - current) > 1)
                        {
                            height_map[ny][nx] = current + ((neighbor > current) ? 1 : -1);
                        }
                    }
                }
            }
        }
    }
}

// Height map generation
int** create_height_array(unsigned width, unsigned height)
{
    // Allocate 2D array
    int** height_map = new int* [height];
    for (unsigned i = 0; i < height; ++i)
    {
        height_map[i] = new int[width];
    }

    float scale = 0.03f;                    // Adjust for different terrain features
    int octaves = 6;                        // Number of noise layers
    float persistence = 0.03f;              // Persistence controls smoothness

    for (unsigned y = 0; y < height; ++y)
    {
        for (unsigned x = 0; x < width; ++x)
        {
            float noise_value = perlinNoise(x * scale, y * scale, octaves, persistence);

            // Map noise to terrain types
            if (noise_value < -0.25f)
            {
                height_map[y][x] = -2;      // Deep sea
            }
            else if (noise_value <= -0.1f)
            {
                height_map[y][x] = -1;      // Shallow sea
            }
            else if (noise_value <= 0.0f)
            {
                height_map[y][x] = 0;       // Sand
            }
            else if (noise_value < 0.1f)
            {
                height_map[y][x] = 1;       // Low land
            }
            else if (noise_value < 0.2f)
            {
                height_map[y][x] = 2;       // High land
            }
            else if (noise_value < 0.35f)
            {
                height_map[y][x] = 3;       // Low mountains
            }
            else
            {
                height_map[y][x] = 4;       // High mountains
            }
        }
    }

    // Enforce gradual transitions
    enforce_gradual_transitions(height_map, width, height);

    return height_map;
}

// Function to 'flatten' the 2D array to 1D for CUDA kernel usages.
int* flatten_height_array(int** arr, int width, int height)
{
    int* flat_arr = new int[width * height];
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            flat_arr[h * width + w] = arr[h][w];
        }
    }
    return flat_arr;
}