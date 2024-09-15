#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include "rastByTri.h"
#include "imageWriter.h"
#define WIDTH 300
#define HEIGHT 300
#ifndef DEBUG
#define DEBUG 0
#endif

void parseTriPair(const std::string &str, float &v1, float &v2, float &v3)
{
	//std::cout << "parsing " << str << std::endl;
	std::stringstream ss;
	ss << str;
	//int tmp;
	/*std::cout << "parsed ";
	for(; ss >> tmp;)
	{
		std::cout  << tmp << " ";
		if(ss.peek() == ',')
			ss.ignore();
	}
	std::cout << std::endl;*/
	if(!(ss >> v1))
	{
		std::cerr << "Could not parse values from triangle input file to integers" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(ss.peek() == ',')
		ss.ignore();
	if(!(ss >> v2))
	{
		std::cerr << "Could not parse values from triangle input file to integers" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(ss.peek() == ',')
		ss.ignore();
	if(!(ss >> v3))
	{
		std::cerr << "Could not parse values from triangle input file to integers" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void readTriangles(thrust::device_vector<thrust::tuple<float,float,float>> &p1,
		thrust::device_vector<thrust::tuple<float,float,float>> &p2,
		thrust::device_vector<thrust::tuple<float,float,float>> &p3,
		thrust::device_vector<thrust::tuple<char,char,char>> &color,
		int &numTri, char *filename)
{
	std::cout << "Start Read Triangles" << std::endl;
	std::ifstream fin(filename);
	std::stringstream ss;
	std::string l1, l2, l3, l4;
	getline(fin, l1);
	ss << l1;
	ss >> numTri;
	p1.resize(numTri);
	p2.resize(numTri);
	p3.resize(numTri);
	color.resize(numTri);
	std::cout << numTri << " Triangles" << std::endl;
	for(int i = 0; i < numTri; i++)
	{
		if(i % 10000 == 0)
			std::cout << "Parsing Triangle " << i << std::endl;
		if(!getline(fin, l1) || !getline(fin, l2) || !getline(fin, l3) || !getline(fin, l4))
		{
			std::cout<<"Not enough values in "<<filename<<" for "<<numTri<<" triangles!" <<std::endl;
			exit(EXIT_FAILURE);
		}
		float v1, v2, v3;
		//std::cout << "color " << l1 << std::endl;
		parseTriPair(l1, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		color[i] = thrust::make_tuple<char,char,char>((char)v1,(char)v2,(char)v3);
		//std::cout << "p1 " << l2 << std::endl;
		parseTriPair(l2, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		p1[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);
		//std::cout << "p2 " << l3 << std::endl;
		parseTriPair(l3, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		p2[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);
		//std::cout << "p3 " << l4 << std::endl;
		parseTriPair(l4, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		p3[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);

	}
	fin.close();
}
	

int main(int argc, char **argv)
{
	if(argc < 3)
	{
		std::cerr << "USAGE: rast <input> <output> " << std::endl;
		exit(EXIT_FAILURE);
	}
#if DEBUG > 0
	std::cout << "initialize triangles" << std::endl;
#endif
	thrust::device_vector<thrust::tuple<float, float, float>> p1;
	thrust::device_vector<thrust::tuple<float, float, float>> p2;
	thrust::device_vector<thrust::tuple<float, float, float>> p3;
	thrust::device_vector<thrust::tuple<char, char, char>> color;

	/*p1[0] = thrust::make_tuple(0,0,0);
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
	//color[2] = thrust::make_tuple(0,255,0);*/

	std::cout << "Start Main" << std::endl;

	int numTri;
	readTriangles(p1, p2, p3, color, numTri, argv[1]);

	std::cout << "Finished Read Triangles" << std::endl;

	Image final_image;
	initImage(&final_image, WIDTH, HEIGHT);

	std::cout << "Finished Initialize Image" << std::endl;

	RasterizeTriangles(p1, p2, p3, color, numTri, WIDTH, HEIGHT, final_image);

	std::cout << "Finished Rasterize Triangles" << std::endl;

	writeImage(&final_image, argv[2]);
	//char *col = final_image.data;
	//for(int i = 0; i < 60; i+=3)
	//{
	//	std::cout<<(int)col[i]<<","<<(int)col[i+1]<<","<<(int)col[i+2]<<std::endl;
	//}

	std::cout << "Finished Write Image" << std::endl;
		
	freeImage(&final_image);

	std::cout << "Program End" << std::endl;
}
