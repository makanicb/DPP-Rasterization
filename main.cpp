#include <thrust/device_vector.h>
#include <iostream>
#include "rastByTri.h"
#include "imageWriter.h"
#define WIDTH 300
#define HEIGHT 300

int main(int argc, char **argv)
{
	std::cout << "initialize triangles" << std::endl;
	thrust::device_vector<thrust::tuple<float, float, float>> p1(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p2(2);
	thrust::device_vector<thrust::tuple<float, float, float>> p3(2);
	thrust::device_vector<thrust::tuple<char, char, char>> color(2);

	p1[0] = thrust::make_tuple(0,0,0);
	p2[0] = thrust::make_tuple(5,5,0);
	p3[0] = thrust::make_tuple(10,0,0);
	p1[1] = thrust::make_tuple(10,0,-0.5);
	p2[1] = thrust::make_tuple(15,15,-0.5);
	p3[1] = thrust::make_tuple(20,0,-0.5);
	//p1[2] = thrust::make_tuple(20,0,-1);
	//p2[2] = thrust::make_tuple(25,25,-1);
	//p3[2] = thrust::make_tuple(30,0,-1);

	color[0] = thrust::make_tuple(255,0,0);
	color[1] = thrust::make_tuple(0,0,255);
	//color[2] = thrust::make_tuple(0,255,0);

	int numTri = 2;

	Image final_image;
	initImage(&final_image, WIDTH, HEIGHT);
	RasterizeTriangles(p1, p2, p3, color, numTri, WIDTH, HEIGHT, final_image);

	if(argc == 2)
	{
		writeImage(&final_image, argv[1]);
	}
	//char *col = final_image.data;
	//for(int i = 0; i < 60; i+=3)
	//{
	//	std::cout<<(int)col[i]<<","<<(int)col[i+1]<<","<<(int)col[i+2]<<std::endl;
	//}
		
	freeImage(&final_image);
}
