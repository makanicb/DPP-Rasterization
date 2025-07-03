#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/Types.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>

#include "rastByTri.h"
#include "imageWriter.h"
#include "readSTL.h"

#ifndef DEBUG
#define DEBUG 0
#endif

void parseTriPair(const std::string &str, float &v1, float &v2, float &v3)
{
	//std::cout << "parsing " << str << std::endl;
	/*std::stringstream ss;
	ss << str;
	//int tmp;
	
	std::cout << "parsed ";
	for(; ss >> tmp;)
	{
		std::cout  << tmp << " ";
		if(ss.peek() == ',')
			ss.ignore();
	}
	std::cout << std::endl;
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
	}*/
	char *temp_line = new char [str.length() + 1];
	strcpy(temp_line, str.c_str());
	char *tok = strtok(temp_line, ",");
	//std::cout << tok << ",";
	v1 = atoi(tok);
	tok = strtok(NULL, ",");
	//std::cout << tok << ",";
	v2 = atoi(tok);
	tok = strtok(NULL, ",");
	//std::cout << tok << ",";
	v3 = atoi(tok);
}

void readTriangles(viskores::cont::ArrayHandle<viskores::Vec3f> &p1,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p2,
		viskores::cont::ArrayHandle<viskores::Vec3f> &p3,
		viskores::cont::ArrayHandle<viskores::Vec3ui_8> &color,
		int &numTri, char *filename, int &width, int &height)
{
	std::cout << "Start Read Triangles" << std::endl;
	std::ifstream fin(filename);
	std::stringstream ss;
	std::string l1, l2, l3, l4;
	getline(fin, l1);
	ss << l1;
	ss >> numTri;
	//resize the output vectors
	p1.Allocate(numTri);
	p2.Allocate(numTri);
	p3.Allocate(numTri);
	color.Allocate(numTri);
	//create reading vectors
	//p1
	auto p1_Writer = p1.WritePortal();
	//p2
	auto p2_Writer = p2.WritePortal();
	//p3
	auto p3_Writer = p3.WritePortal();
	//color
	auto color_Writer = color.WritePortal();
	std::cout << numTri << " Triangles" << std::endl;
	//parse file
	//read contents into linear vectors
	//variables to record minimum x and y values
	int lowx = 0;
	int lowy = 0;
	for(int i = 0; i < numTri; i++)
	{
		//if(i % 10000 == 0)
			//std::cout << "Parsing Triangle " << i << std::endl;
		if(!getline(fin, l1) || !getline(fin, l2) || !getline(fin, l3) || !getline(fin, l4))
		{
			std::cout<<"Not enough values in "<<filename<<" for "<<numTri<<" triangles!" <<std::endl;
			exit(EXIT_FAILURE);
		}
		float v1, v2, v3;
		//std::cout << "color " << l1 << std::endl;
		parseTriPair(l1, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		//color[i] = thrust::make_tuple<char,char,char>((char)v1,(char)v2,(char)v3);
		color_Writer.Set(i, viskores::make_Vec(char(v1), char(v2), char(v3)));
		//std::cout << "p1 " << l2 << std::endl;
		parseTriPair(l2, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		//p1[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);
		p1_Writer.Set(i, viskores::make_Vec(v1, v2, v3));
		width = std::max(width, (int) v1 + 1); //update width
		height = std::max(height, (int) v2 + 1); //update height
		lowx = std::min(lowx, (int) v1); //update lowx
		lowy = std::min(lowy, (int) v2); //update lowy
		//std::cout << "p2 " << l3 << std::endl;
		parseTriPair(l3, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		//p2[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);
		p2_Writer.Set(i, viskores::make_Vec(v1, v2, v3));
		width = std::max(width, (int) v1 + 1); //update width
		width = std::max(width, (int) v1 + 1);
		height = std::max(height, (int) v2 + 1);
		lowx = std::min(lowx, (int) v1);
		lowy = std::min(lowy, (int) v2);
		//std::cout << "p3 " << l4 << std::endl;
		parseTriPair(l4, v1, v2, v3);
		//std::cout <<"parsed "<< v1 << ", " << v2 << ", " << v3 << std::endl;
		//p3[i] = thrust::make_tuple<float,float,float>(v1,v2,v3);
		p3_Writer.Set(i, viskores::make_Vec(v1, v2, v3));
		width = std::max(width, (int) v1 + 1);
		height = std::max(height, (int) v2 + 1);
		lowx = std::min(lowx, (int) v1);
		lowy = std::min(lowy, (int) v2);

	}
	fin.close();

	/*
	auto p1b = thrust::make_zip_iterator(thrust::make_tuple(p11.begin(), p12.begin(), p13.begin()));
	auto p1e = thrust::make_zip_iterator(thrust::make_tuple(p11.end(), p12.end(), p13.end()));
	thrust::copy(p1b, p1e, p1.begin());
	auto p2b = thrust::make_zip_iterator(thrust::make_tuple(p21.begin(), p22.begin(), p23.begin()));
	auto p2e = thrust::make_zip_iterator(thrust::make_tuple(p21.end(), p22.end(), p23.end()));
	thrust::copy(p2b, p2e, p2.begin());
	auto p3b = thrust::make_zip_iterator(thrust::make_tuple(p31.begin(), p32.begin(), p33.begin()));
	auto p3e = thrust::make_zip_iterator(thrust::make_tuple(p31.end(), p32.end(), p33.end()));
	thrust::copy(p3b, p3e, p3.begin());
	auto cb = thrust::make_zip_iterator(thrust::make_tuple(c1.begin(), c2.begin(), c3.begin()));
	auto ce = thrust::make_zip_iterator(thrust::make_tuple(c1.end(), c2.end(), c3.end()));
	thrust::copy(cb, ce, color.begin());
	*/
	width -= lowx;
	height -= lowy;
}

void convertVecToTupleFloat(const viskores::cont::ArrayHandle<viskores::Vec3f> &in,
		viskores::cont::ArrayHandle<thrust::tuple<float, float, float>> &out)
{
	viskores::Id len = in.GetNumberOfValues();
	auto in_read = in.ReadPortal();

	out.Allocate(len);
	auto out_write = out.WritePortal();

	for(viskores::Id i = 0; i < len; i++)
	{
		viskores::Vec3f vec = in_read.Get(i);
		out_write.Set(i, thrust::make_tuple(vec[0], vec[1], vec[2]));
	}
}
	
void convertVecToTupleChar(const viskores::cont::ArrayHandle<viskores::Vec3ui_8> &in,
		viskores::cont::ArrayHandle<thrust::tuple<char, char, char>> &out)
{
	viskores::Id len = in.GetNumberOfValues();
	auto in_read = in.ReadPortal();

	out.Allocate(len);
	auto out_write = out.WritePortal();

	for(viskores::Id i = 0; i < len; i++)
	{
		viskores::Vec3f vec = in_read.Get(i);
		out_write.Set(i, thrust::make_tuple((char)vec[0], (char)vec[1], (char)vec[2]));
	}
}

int main(int argc, char **argv)
{
	//initialize viskores
	viskores::cont::Initialize(argc, argv, viskores::cont::InitializeOptions::AddHelp);

	if(argc < 3)
	{
		std::cerr << "USAGE: rast <input> <output> " << std::endl;
		exit(EXIT_FAILURE);
	}

	//create width and height variables
	int WIDTH = 300;
	int HEIGHT = 300;

#if DEBUG > 0
	std::cout << "initialize triangles" << std::endl;
#endif
	viskores::cont::ArrayHandle<viskores::Vec3f> p1;
	viskores::cont::ArrayHandle<viskores::Vec3f> p2;
	viskores::cont::ArrayHandle<viskores::Vec3f> p3;
	viskores::cont::ArrayHandle<viskores::Vec3ui_8> color;

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

#if DEBUG > 0
	std::cout << "Start Main" << std::endl;
#endif

	int numTri;
	char *fileType = argv[1];
	for(; (*fileType) != '.'; fileType++);
	fileType++;
#if DEBUG > 1
	std::cout << fileType << std::endl;
#endif
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp1; 
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp2;
	viskores::cont::ArrayHandle<thrust::tuple<float,float,float>> vp3;
	viskores::cont::ArrayHandle<thrust::tuple<char,char,char>> vcolor;

	if(strcmp(fileType, "tri") == 0)
	{
		readTriangles(p1, p2, p3, color, numTri, argv[1], WIDTH, HEIGHT);
		convertVecToTupleFloat(p1, vp1);
		convertVecToTupleFloat(p2, vp2);
		convertVecToTupleFloat(p2, vp2);
		convertVecToTupleChar(color, vcolor);
	}
	else if(strcmp(fileType, "stl") == 0)
		numTri = readTriFromBinarySTL(vp1, vp2, vp3, vcolor, argv[1], WIDTH, HEIGHT);
	else
		return -1;

	/*
	std::cout << "P1 LENGTH: " << vp1.GetNumberOfValues() << std::endl;
	std::cout << "P2 LENGTH: " << vp2.GetNumberOfValues() << std::endl;
	std::cout << "P3 LENGTH: " << vp3.GetNumberOfValues() << std::endl;
	std::cout << "COLOR LENGTH: " << vcolor.GetNumberOfValues() << std::endl;
	*/

#if DEBUG > 0
	std::cout << "Finished Read Triangles" << std::endl;
#if DEBUG > 1	
	std::cout << "width: " << WIDTH << " height: " << HEIGHT << std::endl;

	std::cout << "Triangles: " << numTri << std::endl;
#endif
#endif
	Image final_image;
	initImage(&final_image, WIDTH, HEIGHT);
#if DEBUG > 0
	std::cout << "Finished Initialize Image" << std::endl;
#endif
	RasterizeTriangles(vp1, vp2, vp3, vcolor, numTri, WIDTH, HEIGHT, final_image);

#if DEBUG > 0
	std::cout << "Finished Rasterize Triangles" << std::endl;
#endif

	writeImage(&final_image, argv[2]);
	//char *col = final_image.data;
	//for(int i = 0; i < 60; i+=3)
	//{
	//	std::cout<<(int)col[i]<<","<<(int)col[i+1]<<","<<(int)col[i+2]<<std::endl;
	//}

#if DEBUG > 0
	std::cout << "Finished Write Image" << std::endl;
#endif
		
	freeImage(&final_image);

#if DEBUG > 0
	std::cout << "Program End" << std::endl;
#endif
}
