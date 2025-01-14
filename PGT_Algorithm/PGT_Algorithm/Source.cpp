#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

const int RENDER_WIDTH = 160;
const int RENDER_HEIGHT = 90;

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

int** create_height_array(unsigned width, unsigned height)
{
    int max = 30;
    int min = -10;
    int range = max - min + 1;
    int diff = 4;

    int** arr = new int* [height];
    for (int h = 0; h < height; h++)
    {
        arr[h] = new int[width];
        for (int w = 0; w < width; w++)
        {
            arr[h][w] = rand() % range + min;
        }
    }

    for (unsigned h = 0; h < height; h++)
    {
        for (unsigned w = 0; w < width; w++)
        {
            if (h > 0)
            {
                if (arr[h][w] > arr[h - 1][w] + diff)
                    arr[h][w] = arr[h - 1][w] + diff;
                else if (arr[h][w] < arr[h - 1][w] - diff)
                    arr[h][w] = arr[h - 1][w] - diff;
            }
            if (w > 0)
            {
                if (arr[h][w] > arr[h][w - 1] + diff)
                    arr[h][w] = arr[h][w - 1] + diff;
                else if (arr[h][w] < arr[h][w - 1] - diff)
                    arr[h][w] = arr[h][w - 1] - diff;
            }
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

    // Copy the color buffer back to the host
    Uint32* displayPixels = new Uint32[SCREEN_WIDTH * SCREEN_HEIGHT]();
    copy_color_buffer_to_host(displayPixels, SCREEN_WIDTH, SCREEN_HEIGHT);

    bool IsRunning = true;
    SDL_Event event;

    // Main running cycle
    while (IsRunning)
    {
        // Process SDL events
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
            {
                IsRunning = false;
            }
        }

        // Update the height map data on the device
        copy_height_map_to_device(flat_height_map, SCREEN_WIDTH, SCREEN_HEIGHT);

        // Launch the kernel
        launch_kernel(SCREEN_WIDTH, SCREEN_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT);

        // Copy the processed color buffer from the device to the host
        copy_color_buffer_to_host(displayPixels, RENDER_WIDTH, RENDER_HEIGHT);

        // Clear the renderer
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Render the scaled-up terrain
        for (int y = 0; y < RENDER_HEIGHT; ++y)
        {
            for (int x = 0; x < RENDER_WIDTH; ++x)
            {
                Uint32 color = displayPixels[y * RENDER_WIDTH + x];

                SDL_Rect pixelRect = {
                    x * (SCREEN_WIDTH / RENDER_WIDTH),  // Scale X
                    y * (SCREEN_HEIGHT / RENDER_HEIGHT), // Scale Y
                    (SCREEN_WIDTH / RENDER_WIDTH),       // Pixel width
                    (SCREEN_HEIGHT / RENDER_HEIGHT)      // Pixel height
                };

                SDL_SetRenderDrawColor(renderer,
                    (color >> 16) & 0xFF,  // Red
                    (color >> 8) & 0xFF,   // Green
                    color & 0xFF,         // Blue
                    (color >> 24) & 0xFF); // Alpha

                SDL_RenderFillRect(renderer, &pixelRect);
            }
        }

        // Present the frame
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
