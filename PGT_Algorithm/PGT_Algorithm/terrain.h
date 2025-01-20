#ifndef TERRAIN_H
#define TERRAIN_H

#include <vector>

extern int global_seed;

float perlinNoise(float x, float y, int octaves, float persistence);
int** create_height_array(unsigned width, unsigned height);
int* flatten_height_array(int** arr, int width, int height);

void enforce_gradual_transitions(int** height_map, unsigned width, unsigned height);
void seed_rng();

#endif // TERRAIN_H