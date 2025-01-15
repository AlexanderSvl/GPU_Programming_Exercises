#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <SDL.h>
#include <cuda_runtime.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

const int RENDER_WIDTH = 240;
const int RENDER_HEIGHT = 120;

const int BUTTON_WIDTH = 60;
const int BUTTON_HEIGHT = 50;

// Connection with the .cu file, containing the kernel. 
extern "C"
{
    void init_device_height_map(int width, int height);
    void free_device_height_map();
    void copy_height_map_to_device(const int* host_height_map, int width, int height);
    void launch_kernel(int width, int height, int render_width, int render_height);
    void copy_color_buffer_to_host(Uint32* host_buffer, int width, int height);
}

int fade(int t)
{
    return (int)(t * t * t * (t * (t * 6 - 15) + 10));
}

int lerp(int a, int b, int t)
{
    return a + t * (b - a);
}

int grad(int hash, int x, int y)
{
    int h = hash & 15;
    int u = h < 8 ? x : y;
    int v = h < 4 ? y : (h == 12 || h == 14 ? x : 0);
    return (h & 1 ? -u : u) + (h & 2 ? -v : v);
}

int noise(int x, int y)
{
    int p = x + y * 57; // A prime number to shuffle the coordinates
    int n = (p << 13) ^ p;
    return (1.0 - (n * (n * n * 15731 + 789221) + 1376312589 & 0x7fffffff) / 1073741824.0f) * 2 - 1;
}

// Perlin noise function (2D)
double perlin_noise(double x, double y)
{
    int X = (int)std::floor(x) & 255;
    int Y = (int)std::floor(y) & 255;
    double dx = x - std::floor(x);
    double dy = y - std::floor(y);
    double u = fade(dx);
    double v = fade(dy);

    int a = noise(X, Y);
    int b = noise(X + 1, Y);
    int c = noise(X, Y + 1);
    int d = noise(X + 1, Y + 1);

    double result = lerp(lerp(a, b, u), lerp(c, d, u), v);
    return result;
}

// Map the height value to the terrain type with more gradual transitions
int map_to_terrain(double value)
{
    if (value <= 0.0) return -2;  // Deep sea (dark blue)
    if (value <= 0.5) return -1;  // Shallow sea (light blue)
    if (value <= 1.5) return 0;   // Sand (yellow)
    if (value <= 2.0) return 1;   // Low land (light green)
    if (value <= 3.0) return 2;   // High land (dark green)
    if (value <= 3.8) return 3;   // Low mountain (grey)
    return 4;  // High mountain (white)
}

// Ensure gradual transitions between neighboring values (+-1)
double enforce_gradual_transition(double current, double next)
{
    if (next > current + 1) return current + 1; // Don't let the change be more than +1
    if (next < current - 1) return current - 1; // Don't let the change be more than -1
    return next;  // Allow gradual change
}

// Enforce gradual transition in all directions (left, right, up, down)
void enforce_gradual_transitions(int** arr, unsigned width, unsigned height)
{
    bool changesMade = true;
    while (changesMade)
    {
        changesMade = false;
        // Horizontal and vertical checks
        for (unsigned h = 0; h < height; h++)
        {
            for (unsigned w = 0; w < width; w++)
            {
                // Check left (horizontal)
                if (w > 0)
                {
                    int adjusted = enforce_gradual_transition(arr[h][w - 1], arr[h][w]);
                    if (arr[h][w] != adjusted)
                    {
                        arr[h][w] = adjusted;
                        changesMade = true;
                    }
                }
                // Check up (vertical)
                if (h > 0)
                {
                    int adjusted = enforce_gradual_transition(arr[h - 1][w], arr[h][w]);
                    if (arr[h][w] != adjusted)
                    {
                        arr[h][w] = adjusted;
                        changesMade = true;
                    }
                }
                // Check right (horizontal)
                if (w < width - 1)
                {
                    int adjusted = enforce_gradual_transition(arr[h][w + 1], arr[h][w]);
                    if (arr[h][w] != adjusted)
                    {
                        arr[h][w] = adjusted;
                        changesMade = true;
                    }
                }
                // Check down (vertical)
                if (h < height - 1)
                {
                    int adjusted = enforce_gradual_transition(arr[h + 1][w], arr[h][w]);
                    if (arr[h][w] != adjusted)
                    {
                        arr[h][w] = adjusted;
                        changesMade = true;
                    }
                }
            }
        }
    }
}

// Generate the heightmap with Perlin noise, octaves, and gradual transitions
int** create_height_array(unsigned width, unsigned height)
{
    // Seed random number generator
    std::srand(std::time(0));

    // Initialize 2D array for terrain heights
    int** arr = new int* [height];
    for (unsigned i = 0; i < height; i++)
        arr[i] = new int[width];

    // Parameters for Perlin noise with octaves
    double frequency = 0.04;  // Lower frequency for smoother, larger features
    double amplitude = 5.0;   // Amplitude for larger variations with smooth transitions
    int octaves = 6;          // More octaves to add finer detail while keeping smoothness

    // Step 1: Initialize with base random values (random offsets)
    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            // Initialize terrain with random base heights between -2 and 4
            arr[h][w] = rand() % 7 - 2;
        }
    }

    // Step 2: Apply multiple octaves of Perlin noise to adjust the terrain
    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            double final_value = 0.0;
            double current_frequency = frequency;
            double current_amplitude = amplitude;

            // Sum up multiple octaves
            for (int octave = 0; octave < octaves; octave++)
            {
                double perlin_value = perlin_noise(w * current_frequency, h * current_frequency);

                // Normalize the Perlin value to [0, 1]
                perlin_value = (perlin_value + 1) / 2.0;

                // Scale the Perlin value according to the current amplitude
                perlin_value *= current_amplitude;

                // Add the Perlin value for this octave
                final_value += perlin_value;

                // Increase frequency and decrease amplitude for next octave
                current_frequency *= 2.0;  // Double the frequency for finer detail
                current_amplitude *= 0.5; // Halve the amplitude for each octave
            }

            // Apply the final value to the terrain array
            arr[h][w] += final_value;

            // Map the height value to a terrain type (from -2 to 4)
            arr[h][w] = map_to_terrain(arr[h][w]);
        }
    }

    // Step 3: Enforce gradual transition in all directions (up, down, left, right, diagonal)
    enforce_gradual_transitions(arr, width, height);

    // Step 4: Redistribute the terrain values to fit within the desired range [-2, 4]
    double min_val = 9999.0;
    double max_val = -9999.0;

    // Find the min and max values in the heightmap
    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            double value = arr[h][w];
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
        }
    }

    // Normalize the values to fit within [-2, 4]
    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            double normalized_value = (arr[h][w] - min_val) / (max_val - min_val); // Normalize to [0, 1]
            arr[h][w] = -2 + (normalized_value * 6); // Redistribute to [-2, 4]
        }
    }

    return arr;
}

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

int main(int argc, char* argv[])
{
    // SDL initialization
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        SDL_Log("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Terrain Visualization",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    if (!window)
    {
        SDL_Log("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        SDL_Log("Renderer could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        SCREEN_WIDTH, SCREEN_HEIGHT);

    if (!texture)
    {
        SDL_Log("Texture could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Initialize CUDA resources
    init_device_height_map(SCREEN_WIDTH, SCREEN_HEIGHT);

    // Create height map and flatten it
    int** height_map = create_height_array(SCREEN_WIDTH, SCREEN_HEIGHT);
    int* flat_height_map = flatten_height_array(height_map, SCREEN_WIDTH, SCREEN_HEIGHT);

    Uint32* displayPixels = new Uint32[SCREEN_WIDTH * SCREEN_HEIGHT]();
    copy_color_buffer_to_host(displayPixels, SCREEN_WIDTH, SCREEN_HEIGHT);

    bool IsRunning = true;
    SDL_Event event;

    while (IsRunning)
    {
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
            {
                IsRunning = false;
            }
        }

        // Update the height map data on the device
        copy_height_map_to_device(flat_height_map, SCREEN_WIDTH, SCREEN_HEIGHT);
        launch_kernel(SCREEN_WIDTH, SCREEN_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT);

        copy_color_buffer_to_host(displayPixels, RENDER_WIDTH, RENDER_HEIGHT);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Render the scaled-up terrain
        for (int y = 0; y < RENDER_HEIGHT; ++y)
        {
            for (int x = 0; x < RENDER_WIDTH; ++x)
            {
                Uint32 color = displayPixels[y * RENDER_WIDTH + x];

                SDL_Rect pixelRect = {
                    x * (SCREEN_WIDTH / RENDER_WIDTH),      // Scale X
                    y * (SCREEN_HEIGHT / RENDER_HEIGHT),    // Scale Y
                    (SCREEN_WIDTH / RENDER_WIDTH),          // Pixel width
                    (SCREEN_HEIGHT / RENDER_HEIGHT)         // Pixel height
                };

                SDL_SetRenderDrawColor(renderer,
                    (color >> 16) & 0xFF,                   // Red
                    (color >> 8) & 0xFF,                    // Green
                    color & 0xFF,                           // Blue
                    (color >> 24) & 0xFF);                  // Alpha

                SDL_RenderFillRect(renderer, &pixelRect);
            }
        }

        SDL_RenderPresent(renderer);
    }

    // Cleanup
    delete[] displayPixels;
    delete[] flat_height_map;

    for (int h = 0; h < SCREEN_HEIGHT; h++)
    {
        delete[] height_map[h];
    }

    delete[] height_map;

    free_device_height_map();

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
