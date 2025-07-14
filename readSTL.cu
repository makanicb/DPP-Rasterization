#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>
#include <cmath>
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
	unsigned int subdivisions = 1;
	unsigned int subdividedNumTri = numTri * pow(4, subdivisions); 
	//resize arrays
	p1.Allocate(subdividedNumTri);
	p2.Allocate(subdividedNumTri);
	p3.Allocate(subdividedNumTri);
	color.Allocate(subdividedNumTri);
	//create writers
	auto p1_Writer = p1.WritePortal();
	auto p2_Writer = p2.WritePortal();
	auto p3_Writer = p3.WritePortal();
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
		//scale vertices
		for(int j = 0; j < 3; j++)
		{
			v1[j] *= scale;
			v2[j] *= scale;
			v3[j] *= scale;
		}
		//process buffers
		width = std::max(width, std::max((int) v1[0], std::max((int) v2[0], (int) v3[0])) + 1);
		height = std::max(height, std::max((int) v1[1], std::max((int) v2[1], (int) v3[1])) + 1);
		lowx = std::min(lowx, std::min((int) v1[0], std::min((int) v2[0], (int) v3[0])));
		lowy = std::min(lowy, std::min((int) v1[1], std::min((int) v2[1], (int) v3[1])));
	}
	// Move corner to origin
	width -= lowx;
	height -= lowy;
	//std::cout << "W: " << width << " H: " << height << std::endl;
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
		//scale vertices
		for(int j = 0; j < 3; j++)
		{
			v1[j] *= scale;
			v2[j] *= scale;
			v3[j] *= scale;
		}
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

	// Subdivide
	unsigned int curNumTri = i;
	for (int j = 0; j < 1; j++)
	{
		// Create temporary arrays to subdivide into
		viskores::cont::ArrayHandle<viskores::Vec3f> tmp_p1;
		viskores::cont::ArrayHandle<viskores::Vec3f> tmp_p2;
		viskores::cont::ArrayHandle<viskores::Vec3f> tmp_p3;
		viskores::cont::ArrayHandle<viskores::Vec3ui_8> tmp_color;

		// Copy point arrays
		tmp_p1.Allocate(subdividedNumTri);
		tmp_p2.Allocate(subdividedNumTri);
		tmp_p3.Allocate(subdividedNumTri);
		tmp_color.Allocate(subdividedNumTri);

		// Create readers
		auto p1_Reader = p1.ReadPortal();
		auto p2_Reader = p2.ReadPortal();
		auto p3_Reader = p3.ReadPortal();
		auto color_Reader = color.ReadPortal();

		// Create writers	
		auto tmp1_Writer = tmp_p1.WritePortal();
		auto tmp2_Writer = tmp_p2.WritePortal();
		auto tmp3_Writer = tmp_p3.WritePortal();
		auto tmpC_Writer = tmp_color.WritePortal();

		// Generate subdivisions
		for (unsigned int k = 0; k < curNumTri; k++)
		{
			// Get original positions
			viskores::Vec3f v1 = p1_Reader.Get(k);
			viskores::Vec3f v2 = p2_Reader.Get(k);
			viskores::Vec3f v3 = p3_Reader.Get(k);
			viskores::Vec3ui_8 col = color_Reader.Get(k);

			// Calculate midpoints
			viskores::Vec3f m12 = (v1 + v2) * 0.5f;
			viskores::Vec3f m13 = (v1 + v3) * 0.5f;
			viskores::Vec3f m23 = (v2 + v3) * 0.5f;

			// First triangle
			tmp1_Writer.Set(k, v1);
			tmp2_Writer.Set(k, m12);
			tmp3_Writer.Set(k, m13);
			tmpC_Writer.Set(k, col);

			// Second triangle
			tmp1_Writer.Set(k+1, m12);
			tmp2_Writer.Set(k+1, v2);
			tmp3_Writer.Set(k+1, m23);
			tmpC_Writer.Set(k+1, col);

			// Third triangle
			tmp1_Writer.Set(k+2, m13);
			tmp2_Writer.Set(k+2, m23);
			tmp3_Writer.Set(k+2, v3);
			tmpC_Writer.Set(k+2, col);

			// Fourth triangle
			tmp1_Writer.Set(k+3, m13);
			tmp2_Writer.Set(k+3, m12);
			tmp3_Writer.Set(k+3, m23);
			tmpC_Writer.Set(k+3, col);

		}

		//Copy subdivisions to original vectors
		p1.DeepCopyFrom(tmp_p1);
		p2.DeepCopyFrom(tmp_p2);
		p3.DeepCopyFrom(tmp_p3);
		color.DeepCopyFrom(tmp_color);

		curNumTri *= 4;
	}

	//copy host vectors into device vectors
	//thrust::copy(p1.begin(), p1.end(), hp1.begin());
	return curNumTri;
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
