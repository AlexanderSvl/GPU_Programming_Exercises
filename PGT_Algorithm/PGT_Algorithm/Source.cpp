#include <iostream>
#include <SDL.h>

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;

int main(int argc, char* argv[])
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
		return -1;
	}

	SDL_Window* window = SDL_CreateWindow("Terrain generation",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		SCREEN_WIDTH,
		SCREEN_HEIGHT,
		SDL_WINDOW_SHOWN
	);

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
		SDL_PIXELFORMAT_RGBA8888,
		SDL_TEXTUREACCESS_STREAMING,
		SCREEN_WIDTH, SCREEN_HEIGHT
	);

	if (!texture)
	{
		std::cerr << "Failed to create texture: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	Uint32* host_mem_pixels = new Uint32[SCREEN_WIDTH * SCREEN_HEIGHT]();
	Uint32* device_mem_pixels;

	bool IsRunning = true;
	bool IsMousePressed = false;
    bool IsLeftMouseButtonPressed = false;
    bool IsRightMouseButtonPressed = false;

	SDL_Event event;
	int mouseX = -1, mouseY = -1;

    while (IsRunning)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                IsRunning = false;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                IsMousePressed = true;

                mouseX = event.button.x; 
                mouseY = event.button.y; 

                if (event.button.button == SDL_BUTTON_LEFT)
                {
                    IsLeftMouseButtonPressed = true;
                    IsRightMouseButtonPressed = false;
                }
                else if (event.button.button == SDL_BUTTON_RIGHT)
                {
                    IsRightMouseButtonPressed = true;
                    IsLeftMouseButtonPressed = false; 
                }
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
                IsMousePressed = false;
                IsLeftMouseButtonPressed = false;  
                IsRightMouseButtonPressed = false;
            }
        }

        if (IsMousePressed)
        {
            if (IsLeftMouseButtonPressed)
            {
                for (size_t i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; ++i)
                {
                    host_mem_pixels[i] = 0xFF0000FF;
                }
            }
            else if (IsRightMouseButtonPressed)
            {
                for (size_t i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; ++i)
                {
                    host_mem_pixels[i] = 0x00FF00FF;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; ++i)
            {
                host_mem_pixels[i] = 0x0000FFFF;
            }
        }

        // Update the texture and render
        SDL_UpdateTexture(texture, nullptr, host_mem_pixels, SCREEN_WIDTH * sizeof(Uint32));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}