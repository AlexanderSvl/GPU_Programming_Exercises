#include <iostream>
#include <SDL.h>
#include <cuda_runtime.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

const int BUTTON_WIDTH = 60;
const int BUTTON_HEIGHT = 50;

extern "C"
{
	void launch_kernel(float* terrain, int width, int height);
}

int main(int argc, char* argv[])
{
    // SDL initialization
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        SDL_Log("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        return 1;
    }

    // Window initialization
    SDL_Window* window = SDL_CreateWindow("Button Example",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    if (!window)
    {
        SDL_Log("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Renderer initialization
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        SDL_Log("Renderer could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    bool IsRunning = true;
    SDL_Event event;

    // Main running cycle
    while (IsRunning)
    {
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
            {
                IsRunning = false;
            }

            if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                int buttonX = SCREEN_WIDTH - BUTTON_WIDTH;
                int buttonY = 0;

                // Check if click is in close button boudaries
                if (event.button.x >= buttonX &&
                    event.button.x <= buttonX + BUTTON_WIDTH &&
                    event.button.y >= buttonY &&
                    event.button.y <= buttonY + BUTTON_HEIGHT)
                {
                    IsRunning = false;
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); 
        SDL_RenderClear(renderer);

        SDL_Rect buttonRect = { SCREEN_WIDTH - BUTTON_WIDTH, 0, BUTTON_WIDTH, BUTTON_HEIGHT };
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); 
        SDL_RenderFillRect(renderer, &buttonRect);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        int buttonX = SCREEN_WIDTH - BUTTON_WIDTH;
        int buttonY = 0;
        int buttonRight = buttonX + BUTTON_WIDTH;
        int buttonBottom = buttonY + BUTTON_HEIGHT;

        int padding = 15; 

        int innerX = buttonX + padding;
        int innerY = buttonY + padding;
        int innerRight = buttonRight - padding;
        int innerBottom = buttonBottom - padding;

        SDL_RenderDrawLine(renderer, innerX, innerY, innerRight, innerBottom); // Top-left to bottom-right
        SDL_RenderDrawLine(renderer, innerRight, innerY, innerX, innerBottom); // Top-right to bottom-left

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}