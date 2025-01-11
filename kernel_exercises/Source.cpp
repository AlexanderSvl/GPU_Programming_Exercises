#include <iostream>
#include <SDL.h>

int main(int argc, char* argv[]) 
{
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) 
    {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Get the current display mode (screen width and height)
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

    // Create the window with screen resolution
    SDL_Window* window = SDL_CreateWindow("Centered Rectangle",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        screenWidth, screenHeight, SDL_WINDOW_SHOWN);

    if (!window) 
    {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) 
    {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Define rectangle dimensions
    int rectWidth = 200;
    int rectHeight = 150;

    // Calculate centered position
    SDL_Rect rect = { (screenWidth - rectWidth) / 2, (screenHeight - rectHeight) / 2, rectWidth, rectHeight };

    // Main loop
    bool running = true;
    SDL_Event event;

    while (running) 
    {
        // Event handling
        while (SDL_PollEvent(&event)) 
        {
            if (event.type == SDL_QUIT) 
            {
                running = false;
            }
        }

        // Clear the screen with a black color
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Black
        SDL_RenderClear(renderer);

        // Set the draw color for the rectangle (e.g., blue)
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue

        // Draw the rectangle
        SDL_RenderFillRect(renderer, &rect);

        // Present the rendered content
        SDL_RenderPresent(renderer);
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
