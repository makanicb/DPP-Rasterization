#ifndef READ_STL
#define READ_STL

//#include <thrust/tuple.h>

#include <viskores/cont/ArrayHandle.h>
#include <viskores/Types.h>

int readBinarySTL(char *filename);
unsigned int getNumTriSTL(char *filename);
unsigned int readTriFromBinarySTL(
		viskores::cont::ArrayHandle<viskores::Vec3f> &p1,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p2,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p3,
		viskores::cont::ArrayHandle<viskores::Vec3ui_8> &color,
		char *filename, int &width, int &height,
		int scale);
#endif
