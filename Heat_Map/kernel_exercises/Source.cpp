#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>

const int SCREEN_WIDTH = 1920;  // You can set a larger resolution
const int SCREEN_HEIGHT = 1080;

// Declare CUDA functions
extern "C"
{
    void initHeatmaps(int width, int height);
    void freeHeatmaps();
    void processSurface(int width, int height, int mouseX, int mouseY);
    void copyHeatmapToHost(float* hostHeatmap, int width, int height);
}

// Function to map heat value to RGB color
Uint32 heatToColor(float heat)
{
    // Clamp heat between 0 and 255
    heat = fminf(fmaxf(heat, 0.0f), 255.0f);

    Uint8 red, green, blue;

    if (heat < 128) 
    {
        blue = static_cast<Uint8>(255 - 2 * heat);   
        green = static_cast<Uint8>(2 * heat);       
        red = static_cast<Uint8>(0);                
    }
    else
    {
        blue = static_cast<Uint8>(0);              
        green = static_cast<Uint8>(255 - 2 * (heat - 128));
        red = static_cast<Uint8>(2 * (heat - 128));       
    }

    Uint8 alpha = 255; // Fully opaque
    return (alpha << 24) | (red << 16) | (green << 8) | blue; // ARGB format
}


int main(int argc, char* argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("Heatmap Simulation",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    if (!window)
    {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        SCREEN_WIDTH, SCREEN_HEIGHT);

    if (!renderer || !texture)
    {
        std::cerr << "Failed to create renderer or texture: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Initialize heatmaps
    initHeatmaps(SCREEN_WIDTH, SCREEN_HEIGHT);

    // Host heatmap and display pixels
    float* hostHeatmap = new float[SCREEN_WIDTH * SCREEN_HEIGHT]();
    Uint32* displayPixels = new Uint32[SCREEN_WIDTH * SCREEN_HEIGHT]();

    bool running = true;
    SDL_Event event;
    int mouseX = -1, mouseY = -1;
    bool isMousePressed = false;

    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
                mouseX = event.motion.x;
                mouseY = event.motion.y;

                if (isMousePressed)
                {
                    processSurface(SCREEN_WIDTH, SCREEN_HEIGHT, mouseX, mouseY);
                }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                isMousePressed = true;
                mouseX = event.button.x;
                mouseY = event.button.y;

                processSurface(SCREEN_WIDTH, SCREEN_HEIGHT, mouseX, mouseY);
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
                isMousePressed = false;
            }
        }

        // Copy heatmap data from GPU
        copyHeatmapToHost(hostHeatmap, SCREEN_WIDTH, SCREEN_HEIGHT);

        // Convert heatmap to ARGB pixels
        for (int i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; ++i)
        {
            displayPixels[i] = heatToColor(hostHeatmap[i]);
        }

        // Update texture and render
        SDL_UpdateTexture(texture, nullptr, displayPixels, SCREEN_WIDTH * sizeof(Uint32));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    // Cleanup
    delete[] hostHeatmap;
    delete[] displayPixels;
    freeHeatmaps();

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}