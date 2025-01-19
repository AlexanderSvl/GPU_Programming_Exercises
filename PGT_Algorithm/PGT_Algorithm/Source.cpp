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

// SDL-specific headers
#include <SDL_ttf.h>
#include <SDL.h>

#include "terrain.h";

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

// Function to render text in a semi-transparent dark grey rectangle
void render_text(SDL_Renderer* renderer, TTF_Font* font, const std::string& text, int width, int height)
{
    // Create a surface with the text
    SDL_Color textColor = { 255, 255, 255 };        // White text color
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
        width - (textSurface->w + 2 * padding) - 10,        // 5px padding from the right edge, 5px from the bottom
        height - (textSurface->h + 2 * padding) - 10,       // 5px padding from the bottom edge
        textSurface->w + 2 * padding,                       // width of the background rect including padding
        textSurface->h + 2 * padding                        // height of the background rect including padding
    };

    // Set the blend mode for the background rectangle to allow transparency
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 128);         // Dark grey with transparency
    SDL_RenderFillRect(renderer, &bgRect);

    // Define the rectangle for the text (inside the background)
    SDL_Rect textRect = {
        width - textSurface->w - padding - 10,              // Apply padding for X
        height - textSurface->h - padding - 10,             // Apply padding for Y
        textSurface->w,                                     // Text width (no padding here)
        textSurface->h                                      // Text height (no padding here)
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