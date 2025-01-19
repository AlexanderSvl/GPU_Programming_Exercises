# CUDA kernel exercieses using SDL2

## Terrain Generation Project

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

### Example:
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/69356c44-4276-4bf7-9b68-486a555e9dad" alt="PGT Algorithm visualization" />
  <p style="display: inline-block;">PGT Algorithm visualization</p>
</div>

---
