#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>
/*
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
*/

#include<viskores/cont/ArrayHandle.h>
#include<viskores/Types.h>

#include "readSTL.h"

/*
    readBinarySTL: A function for reading a binary STL file to standard output

    Parameters:
    * filename: a pointer to an array of characters containing the
    	name of the file to read

    Returns 0 on sucess
*/
int readBinarySTL (char *filename)
{
	//open the stl file in read mode
	FILE *f = fopen(filename, "r");
	//create buffers to store the header of the file and the
	//number of triangles
	char header[80];
	unsigned int numTri = 0;
	//read from the file into the buffers
	fread(header, 1, 80, f);
	fread(&numTri, 4, 1, f);
	//print the values in the buffers
	std::cout << header << std::endl;
	std::cout << numTri << std::endl;
	//read the triangles
	//create buffers for the normal vector, position vectors
	//for the three vertices of each triangle, and attribute
	//byte count
	float normal[3];
	float vert1[3];
	float vert2[3];
	float vert3[3];
	short attr;
	for (unsigned int i = 0; i < numTri; i++)
	{
		//read into the buffers
		fread(normal, 4, 3, f);
		fread(vert1, 4, 3, f);
		fread(vert2, 4, 3, f);
		fread(vert3, 4, 3, f);
		fread(&attr, 2, 1, f);
		//print out the values in the buffers
		std::cout << "N\tV1\tV2\tV3" << std::endl;
		for(int j = 0; j < 3; j++)
		{
			std::cout << normal[j] << "\t";
			std::cout << vert1[j] << "\t";
			std::cout << vert2[j] << "\t";
			std::cout << vert3[j] << "\t";
			std::cout << std::endl;
		}
		std::cout << "Attribute Byte Count: " << attr << std::endl;
	}
	//close the file
	fclose(f);
	return 0;
}

/*
    getNumTriSTL: Get the number of triangles encoded within an STL file

    Parameters:
    * filename: a pointer to an array of characters containing the name
    	of the STL file to read

    Returns the number of triangles in the file
*/
unsigned int getNumTriSTL(char *filename)
{
	//open the STL file
	FILE *f = fopen(filename, "r");
	//go to the end of the STL file header
	fseek(f, 80, SEEK_SET);
	//create a buffer to read the number of triangles into
	unsigned int numTri = 0;
	//read the number of triangles into the buffer
	fread(&numTri, 4, 1, f);
	//close the STL file
	fclose(f);
	//return the number of triangles
	return numTri;
}

template<typename WritePortalType>
int getColor(float* norm, WritePortalType &color, int index)
{
	float light[3] = {0, 0, 1};
	float mag = std::sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
	float dot = (norm[0] * light[0] + norm[1] * light[1] + norm[2] * light[2]) / mag;
	dot = std::max(dot, 0.0f);
	color.Set(index, viskores::make_Vec((char)(255 * dot), (char)(255 * dot), (char)(255 * dot)));
	/*if(dot > 1)
	{
		std::cout << std::endl;
		std::cout << norm[0] << ", " << norm[1] << ", " << norm[2] << std::endl;
		std::cout << dot << ", " << mag << std::endl;
	}*/
	//std::cout << 255 * dot << ", " << (int) (255 * dot) << ", " << (int)(unsigned char)(char) (255 * dot) << std::endl;
	return 0;
}

/*
    readTriFromBinarySTL: read the triangles of an STL file into buffers

    Parameters:
    * p1: the address of the buffer to store the positions of the first
    	point of each triangle in
    * p2: the address of the buffer to store the positions of the
    	second point of each triangle in
    * p3: the address of the buffer to store the positions of the third
    	point of each triangle in
    * color: the address of the buffer to store the color of each
    	triangle in
    * filename: the name of the file to read

    Returns the number of triangles read
*/

unsigned int readTriFromBinarySTL(
	viskores::cont::ArrayHandle<viskores::Vec3f> &p1,
	viskores::cont::ArrayHandle<viskores::Vec3f> &p2,
	viskores::cont::ArrayHandle<viskores::Vec3f> &p3,
	viskores::cont::ArrayHandle<viskores::Vec3ui_8> &color,
	char *filename, int &width, int &height, int scale)
{
	//get the number of triangles to read
	unsigned int numTri = getNumTriSTL(filename);
	//resize arrays
	p1.Allocate(numTri);
	p2.Allocate(numTri);
	p3.Allocate(numTri);
	color.Allocate(numTri);
	//create writers
	auto p1_Writer = p1.ReadWritePortal();
	auto p2_Writer = p2.ReadWritePortal();
	auto p3_Writer = p3.ReadWritePortal();
	auto color_Writer = color.WritePortal();
	//open the file
	FILE *f = fopen(filename, "r");
	//go to the start of the triangle information
	fseek(f, 84, SEEK_SET);
	//iterate over the triangles
	unsigned int i = 0;
	//create buffers for each bit of triangle information
	float norm[3];
	float v1[3];
	float v2[3];
	float v3[3];
	short attr;
	//iterate over triangles for max and min values
	//variables for tracking lowest values
	int lowx = std::numeric_limits<int>::max();
	int lowy = std::numeric_limits<int>::max();
	//reset width and height to minimum possible value
	width = std::numeric_limits<int>::min();
	height = std::numeric_limits<int>::min();
	for(; i < numTri; i++)
	{
		//read into buffers
		fread(norm, 4, 3, f);
		fread(v1, 4, 3, f);
		fread(v2, 4, 3, f);
		fread(v3, 4, 3, f);
		fread(&attr, 2, 1, f);
		//process buffers
		width = std::max(width, std::max((int) v1[0], std::max((int) v2[0], (int) v3[0])) + 1);
		height = std::max(height, std::max((int) v1[1], std::max((int) v2[1], (int) v3[1])) + 1);
		lowx = std::min(lowx, std::min((int) v1[0], std::min((int) v2[0], (int) v3[0])));
		lowy = std::min(lowy, std::min((int) v1[1], std::min((int) v2[1], (int) v3[1])));
	}
	// Move corner to origin
	width -= lowx;
	height -= lowy;
	// Scale size
	width *= scale;
	height *= scale;
	//std::cout << "LOWX: " << lowx << " LOWY: " << lowy << std::endl;

	fseek(f, 84, SEEK_SET); //go back to the start of the file

	//iterate over the triangles to read into triangle buffers
	
	//std::cout << "x, y, z" << std::endl;
	for(i = 0; i < numTri; i++)
	{
		//read into buffers
		fread(norm, 4, 3, f);
		fread(v1, 4, 3, f);
		fread(v2, 4, 3, f);
		fread(v3, 4, 3, f);
		fread(&attr, 2, 1, f);
		//process buffers
		//std::cout << v1[0] - lowx << ", " << v1[1] - lowy << ", " << v1[2] << std::endl;
		p1_Writer.Set(i, viskores::make_Vec(v1[0] - lowx, v1[1] - lowy, v1[2]));
		//std::cout << v2[0] - lowx << ", " << v2[1] - lowy << ", " << v2[2] << std::endl;
		p2_Writer.Set(i, viskores::make_Vec(v2[0] - lowx, v2[1] - lowy, v2[2]));
		//std::cout << v3[0] - lowx << ", " << v3[1] - lowy << ", " << v3[2] << std::endl;
		p3_Writer.Set(i, viskores::make_Vec(v3[0] - lowx, v3[1] - lowy, v3[2]));
		getColor(norm, color_Writer, i);
	}
	//std::cout << std::endl;

	// Scale triangles
	for(i = 0; i < numTri; i++)
	{
		p1_Writer.Set(i, p1_Writer.Get(i) * scale);
		p2_Writer.Set(i, p2_Writer.Get(i) * scale);
		p3_Writer.Set(i, p3_Writer.Get(i) * scale);
	}

	//copy host vectors into device vectors
	//thrust::copy(p1.begin(), p1.end(), hp1.begin());
	return i;
}

/*int main(int argc, char **argv)
{
	readBinarySTL(argv[1]);
	unsigned int numTri = getNumTriSTL(argv[1]);
	std::cout << numTri << std::endl;
	thrust::device_vector<thrust::tuple<float,float,float>> p1(numTri);
	thrust::device_vector<thrust::tuple<float,float,float>> p2(numTri);
	thrust::device_vector<thrust::tuple<float,float,float>> p3(numTri);
	thrust::device_vector<thrust::tuple<char,char,char>> color(numTri);
	readTriFromBinarySTL(p1, p2, p3, color, argv[1]);
	thrust::host_vector<thrust::tuple<float,float,float>> temp = p1;
	for(int i = 0; i < temp.size(); i++)
		std::cout << temp[i].get<0>() << "\t";
	std::cout << std::endl;
	return 0;
}*/
