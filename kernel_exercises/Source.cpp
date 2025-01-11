#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

extern "C" 
{
    void processSurface(Uint32* devPixels, 
                        int width, 
                        int height, 
                        int mouseX, 
                        int mouseY);
}

int main(int argc, char* argv[])
{
    // First, we initialize SDL and check if it was created successfully.
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Get the current display mode
    int screenWidth, screenHeight;
    SDL_DisplayMode DM;

    if (SDL_GetCurrentDisplayMode(0, &DM) != 0)
    {
        std::cerr << "Failed to get display mode: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    screenWidth = DM.w;
    screenHeight = DM.h;

    SDL_Window* window = SDL_CreateWindow("Fullscreen Window",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        screenWidth, screenHeight, SDL_WINDOW_SHOWN);

    if (!window)
    {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (!renderer)
    {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        SCREEN_WIDTH, SCREEN_HEIGHT);

    if (!texture)
    {
        std::cerr << "Failed to create texture: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    Uint32* hostPixels = new Uint32[SCREEN_WIDTH * SCREEN_HEIGHT]();
    Uint32* devPixels;

    cudaMalloc(&devPixels, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));
    cudaMemset(devPixels, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

    bool running = true;
    SDL_Event event;
    int mouseX = -1, mouseY = -1;

    bool isMousePressed = false; // Variable to track if mouse is being pressed

    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            // Handle mouse motion events
            else if (event.type == SDL_MOUSEMOTION)
            {
                // Update mouse position
                mouseX = event.motion.x;
                mouseY = event.motion.y;

                // If mouse is being held down, process surface (draw)
                if (isMousePressed)
                {
                    // Call the kernel to process the surface
                    processSurface(devPixels, SCREEN_WIDTH, SCREEN_HEIGHT, mouseX, mouseY);

                    // Copy processed pixels back to host memory
                    cudaMemcpy(hostPixels, devPixels, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);
                }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                isMousePressed = true;

                // Start drawing
                mouseX = event.button.x;
                mouseY = event.button.y;

                processSurface(devPixels, SCREEN_WIDTH, SCREEN_HEIGHT, mouseX, mouseY);
                cudaMemcpy(hostPixels, devPixels, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
                isMousePressed = false;
            }
        }

        // After processing, update the SDL texture and render the image
        SDL_UpdateTexture(texture, nullptr, hostPixels, SCREEN_WIDTH * sizeof(Uint32));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }


    cudaFree(devPixels);
    delete[] hostPixels;

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}