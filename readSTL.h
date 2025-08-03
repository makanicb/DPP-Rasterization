#ifndef READ_STL
#define READ_STL

#include <thrust/device_vector.h>
#include <thrust/tuple.h>

int readBinarySTL(char *filename);
unsigned int getNumTriSTL(char *filename);
unsigned int readTriFromBinarySTL(
		thrust::device_vector<thrust::tuple<float,float,float>> &p1,
		thrust::device_vector<thrust::tuple<float,float,float>> &p2,
		thrust::device_vector<thrust::tuple<float,float,float>> &p3,
		thrust::device_vector<thrust::tuple<char,char,char>> &color,
		char *filename, int &width, int &height,
		int scale, int subdivisions);
#endif
