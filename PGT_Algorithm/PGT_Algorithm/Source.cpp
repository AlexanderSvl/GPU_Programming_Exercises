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

const int RENDER_WIDTH = 480;
const int RENDER_HEIGHT = 270;

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

float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float lerp(float t, float a, float b)
{
    return a + t * (b - a);
}

float grad(int hash, float x, float y)
{
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 4) ? y : ((h == 12) || (h == 14)) ? x : 0;
    return ((h & 1 ? -u : u) + (h & 2 ? -v : v));
}

float perlin_noise(float x, float y, const std::vector<int>& p)
{
    int X = (int)floor(x) & 255;
    int Y = (int)floor(y) & 255;
    float xf = x - floor(x);
    float yf = y - floor(y);
    float u = fade(xf);
    float v = fade(yf);
    int A = p[X] + Y;
    int B = p[X + 1] + Y;
    return lerp(v, lerp(u, grad(p[A], xf, yf), grad(p[B], xf - 1, yf)),
        lerp(u, grad(p[A + 1], xf, yf - 1), grad(p[B + 1], xf - 1, yf - 1)));
}

std::vector<int> generate_permutation_table()
{
    std::vector<int> p(512);
    for (int i = 0; i < 256; i++)
    {
        p[i] = i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < 256; i++)
    {
        int j = dis(gen);
        std::swap(p[i], p[j]);
    }

    for (int i = 0; i < 256; i++)
    {
        p[i + 256] = p[i];
    }
    return p;
}

// Smoothing function
void smooth_height_map(int** arr, unsigned width, unsigned height, int iterations)
{
    for (int iter = 0; iter < iterations; ++iter)
    {
        int** temp = new int* [height];
        for (unsigned h = 0; h < height; ++h)
        {
            temp[h] = new int[width];
        }

        for (unsigned h = 1; h < height - 1; ++h)
        {
            for (unsigned w = 1; w < width - 1; ++w)
            {
                temp[h][w] = (arr[h][w] + arr[h - 1][w] + arr[h + 1][w] +
                    arr[h][w - 1] + arr[h][w + 1]) / 5;
            }
        }

        for (unsigned h = 0; h < height; ++h)
        {
            for (unsigned w = 0; w < width; ++w)
            {
                arr[h][w] = temp[h][w];
            }
        }

        for (unsigned h = 0; h < height; ++h)
        {
            delete[] temp[h];
        }
        delete[] temp;
    }
}

template<typename T>
T clamp(T value, T min, T max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Main function to create height array
int** create_height_array(unsigned width, unsigned height)
{
    int max = 100;    // Max height (for mountains)
    int min = -100;   // Min height (for deep valleys)
    int range = max - min + 1;
    int diff = 25;    // Control fluctuation (greater value leads to less fluctuation)

    std::vector<int> p = generate_permutation_table();

    int** arr = new int* [height];
    for (unsigned h = 0; h < height; h++)
    {
        arr[h] = new int[width];
        for (unsigned w = 0; w < width; w++)
        {
            // Generate Perlin noise values for large-scale terrain features (mountain ranges, valleys)
            float noise_value = perlin_noise(w * 0.1f, h * 0.1f, p); // Larger scale for general features

            // Add mid and high-frequency details for finer granularity
            noise_value += 0.5f * perlin_noise(w * 0.5f, h * 0.5f, p); // Mid scale for smoothing
            noise_value += 0.25f * perlin_noise(w * 2.0f, h * 2.0f, p); // High frequency for detailed features

            // Normalize the noise value to range from 0 to 1
            noise_value = (noise_value + 1.0f) / 2.0f; // Normalize to [0, 1]

            // Map to the desired range (-100 to 100) for terrain height
            arr[h][w] = static_cast<int>((noise_value * range) + min);

            // Ensure heights stay within the min/max boundaries
            arr[h][w] = clamp(arr[h][w], min, max);
        }
    }

    // Apply a mountain boost effect at random spots
    int mountain_boost_threshold = 50; // Elevation value at which we boost mountains
    int mountain_intensity = 20;  // Boost amount to make mountains more pronounced

    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            // Randomly boost areas that are already high enough to be mountains
            if (arr[h][w] > mountain_boost_threshold)
            {
                if (rand() % 100 < 5)  // Small chance of mountain formation (5%)
                {
                    arr[h][w] = std::min(arr[h][w] + mountain_intensity, max);
                }
            }
        }
    }

    // Apply smoothing to the height map to create gradual terrain transitions (this is essential for realism)
    smooth_height_map(arr, width, height, 4);

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
