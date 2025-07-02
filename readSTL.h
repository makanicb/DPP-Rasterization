#ifndef READ_STL
#define READ_STL

#include <thrust/tuple.h>

#include <viskores/cont/ArrayHandle.h>

int readBinarySTL(char *filename);
unsigned int getNumTriSTL(char *filename);
unsigned int readTriFromBinarySTL(
		viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> &p1,
		viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> &p2,
		viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> &p3,
		viskores::cont::ArrayHandle<thrust::tuple<char,char,char>> &color,
		char *filename, int &width, int &height);
#endif
