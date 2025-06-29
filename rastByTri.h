#ifndef RAST_BY_TRI
#define RAST_BY_TRI
#include <thrust/device_vector.h>
#include "imageWriter.h"

void RasterizeTriangles(viskores::cont::ArrayHandle<thrust::tuple<float, float, float>> &p1,
		viskores::cont::ArrayHandle<thrust::tuple<float, float, float>> &p2,
		viskores::cont::ArrayHandle<thrust::tuple<float, float, float>> &p3,
		viskores::cont::ArrayHandle<thrust::tuple<char, char, char>> &color,
		int numTri, int width, int height, Image &final_image);
#endif
