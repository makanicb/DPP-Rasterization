#ifndef RAST_BY_TRI
#define RAST_BY_TRI
//#include <thrust/device_vector.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/Types.h>
#include "imageWriter.h"

void RasterizeTriangles(viskores::cont::ArrayHandle<viskores::Vec3f> &p1,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p2,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p3,
		viskores::cont::ArrayHandle<viskores::Vec3ui_8> &color,
		int numTri, int width, int height, Image &final_image, bool warmup);
#endif
