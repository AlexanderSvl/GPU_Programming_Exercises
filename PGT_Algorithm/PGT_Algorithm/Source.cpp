#define SDL_MAIN_HANDLED

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>

#include <SDL_ttf.h>
#include <SDL.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

const int RENDER_WIDTH = 240;
const int RENDER_HEIGHT = 135;

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

float noise(float x, float y)
{
    int n = static_cast<int>(x + y * 57);
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

// Updated height map generation
int** create_height_array(unsigned width, unsigned height)
{
    // Allocate 2D array
    int** height_map = new int* [height];
    for (unsigned i = 0; i < height; ++i)
    {
        height_map[i] = new int[width];
    }

    float scale = 0.03f; // Adjust for different terrain features
    int octaves = 6;    // Number of noise layers
    float persistence = 0.03f; // Persistence controls smoothness

    for (unsigned y = 0; y < height; ++y)
    {
        for (unsigned x = 0; x < width; ++x)
        {
            float noise_value = perlinNoise(x * scale, y * scale, octaves, persistence);

            // Map noise to terrain types
            if (noise_value < -0.25f)
            {
                height_map[y][x] = -2; // Deep sea
            }
            else if (noise_value <= -0.1f)
            {
                height_map[y][x] = -1; // Shallow sea
            }
            else if (noise_value <= 0.0f)
            {
                height_map[y][x] = 0; // Sand
            }
            else if (noise_value < 0.1f)
            {
                height_map[y][x] = 1; // Low land
            }
            else if (noise_value < 0.2f)
            {
                height_map[y][x] = 2; // High land
            }
            else if (noise_value < 0.35f)
            {
                height_map[y][x] = 3; // Low mountains
            }
            else
            {
                height_map[y][x] = 4; // High mountains
            }
        }
    }

    // Enforce gradual transitions
    enforce_gradual_transitions(height_map, width, height);

    return height_map;
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

// Function to render text in a semi-transparent dark grey rectangle
void render_text(SDL_Renderer* renderer, TTF_Font* font, const std::string& text, int width, int height)
{
    // Create a surface with the text
    SDL_Color textColor = { 255, 255, 255 };  // White text color
    SDL_Surface* textSurface = TTF_RenderText_Solid(font, text.c_str(), textColor);
    if (!textSurface)
    {
        SDL_Log("Unable to create text surface! TTF_Error: %s", TTF_GetError());
        return;
    }

    // Create a texture from the surface
    SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
    if (!textTexture)
    {
        SDL_Log("Unable to create text texture! SDL_Error: %s", SDL_GetError());
        SDL_FreeSurface(textSurface);
        return;
    }

    // Define padding
    const int padding = 5;

    // Define the rectangle for the background (semi-transparent dark grey)
    SDL_Rect bgRect = {
        width - (textSurface->w + 2 * padding) - 10,  // 5px padding from the right edge, 5px from the bottom
        height - (textSurface->h + 2 * padding) - 10, // 5px padding from the bottom edge
        textSurface->w + 2 * padding, // width of the background rect including padding
        textSurface->h + 2 * padding  // height of the background rect including padding
    };

    // Set the blend mode for the background rectangle to allow transparency
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 128);  // Dark grey with transparency
    SDL_RenderFillRect(renderer, &bgRect);

    // Define the rectangle for the text (inside the background)
    SDL_Rect textRect = {
        width - textSurface->w - padding - 10,  // Apply padding for X
        height - textSurface->h - padding - 10, // Apply padding for Y
        textSurface->w,  // Text width (no padding here)
        textSurface->h   // Text height (no padding here)
    };

    // Render the text texture on top of the background
    SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

    // Reset the blend mode to default (so the rest of the rendering isn't affected)
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);

    // Clean up
    SDL_DestroyTexture(textTexture);
    SDL_FreeSurface(textSurface);
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

    if (TTF_Init() == -1)
    {
        SDL_Log("SDL_ttf could not initialize! TTF_Error: %s", TTF_GetError());
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Load a font
    TTF_Font* font = TTF_OpenFont("../fonts/zain.ttf", 24); // Adjust font size and path
    if (!font)
    {
        SDL_Log("Failed to load font! TTF_Error: %s", TTF_GetError());
        SDL_DestroyTexture(texture);
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
    int mouseX = -1, mouseY = -1;

    while (IsRunning)
    {
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
            {
                IsRunning = false;
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
                SDL_GetMouseState(&mouseX, &mouseY);
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

        // Render height at mouse position
        if (mouseX >= 0 && mouseX < SCREEN_WIDTH && mouseY >= 0 && mouseY < SCREEN_HEIGHT)
        {
            int height_value = height_map[mouseY][mouseX];
            std::string height_text = "Height: " + std::to_string(height_value);
            render_text(renderer, font, height_text, SCREEN_WIDTH, SCREEN_HEIGHT);
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