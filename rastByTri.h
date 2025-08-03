#ifndef RAST_BY_TRI
#define RAST_BY_TRI
#include <thrust/device_vector.h>
#include "imageWriter.h"

void RasterizeTriangles(thrust::device_vector<thrust::tuple<float, float, float>> &p1,
		thrust::device_vector<thrust::tuple<float, float, float>> &p2,
		thrust::device_vector<thrust::tuple<float, float, float>> &p3,
		thrust::device_vector<thrust::tuple<char, char, char>> &color,
		int numTri, int width, int height, Image &final_image, bool warmup);
#endif
