# CUDA kernel exercieses using SDL2

## 1. Heat Map Project

### Overview:
This project simulates a simple heat map on the screen where heat intensity increases the longer the mouse stays at one location. It uses a CUDA kernel to simulate heat diffusion and SDL for rendering the graphical interface. The heat propagates from the mouse position, creating a realistic heat spread.

### Features:
- **Real-time Heat Simulation**: Generates heat at mouse position, intensifying as the mouse stays still.
- **CUDA Heat Diffusion**: The heat dissipation algorithm is implemented in CUDA, using parallel computing for efficiency.
- **SDL Rendering**: Visualizes the heat map with interactive mouse input.

### Technologies:
- **CUDA**: Used for the heat diffusion simulation.
- **SDL**: Used for graphical rendering and user input.
- **C++**: The core language for integrating CUDA and SDL.

### How It Works:
- Heat sources are generated when the mouse is pressed and held, with intensity increasing over time.
- The heat dissipation algorithm runs in parallel via CUDA, updating the heat map in real-time.
- SDL renders the updated heat map with color representation based on intensity.

---

## 2. Terrain Generation Project

### Overview:
This project generates realistic terrain maps using Perlin noise, CUDA for parallel computation, and SDL for visualization. It simulates mountain ranges, valleys, oceans, and plains, with smooth transitions between different terrain types.

### Features:
- **Realistic Terrain**: Generates natural-looking mountains, valleys, and oceans.
- **CUDA-based Terrain Computation**: Uses CUDA for efficient terrain generation.
- **Smooth Transitions**: Uses Perlin noise for smooth terrain blending.
- **Real-Time Visualization**: Visualizes terrain in SDL with color-coded regions.

### Technologies:
- **CUDA**: Used for parallel terrain generation.
- **SDL**: Used for terrain rendering and user interaction.
- **C++**: The core language for integrating CUDA and SDL.
- **Perlin Noise**: Used for generating terrain features.

### How It Works:
- A height map is generated using Perlin noise for large and fine-scale terrain features.
- Mountain ranges are formed using a flood-fill technique, and terrain smoothing ensures natural transitions.
- The generated terrain is color-mapped and displayed in real-time using SDL.

---
