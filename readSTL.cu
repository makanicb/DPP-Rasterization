#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>

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
	for (int i = 0; i < numTri; i++)
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
	std::cout << ftell(f) << std::endl;
	//create a buffer to read the number of triangles into
	unsigned int numTri = 0;
	//read the number of triangles into the buffer
	fread(&numTri, 4, 1, f);
	//close the STL file
	fclose(f);
	//return the number of triangles
	return numTri;
}

thrust::tuple<char,char,char> getColor(float* norm)
{
	float light[3] = {0, 0, 1};
	float dot = norm[0] * light[0] + norm[1] * light[1] + norm[2] * light[2];
	dot = max(dot, 0.0);
	return thrust::make_tuple((char)(255 * dot), (char)(255 * dot), (char)(255 * dot));
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
	thrust::device_vector<thrust::tuple<float,float,float>> &p1,
	thrust::device_vector<thrust::tuple<float,float,float>> &p2,
	thrust::device_vector<thrust::tuple<float,float,float>> &p3,
	thrust::device_vector<thrust::tuple<char,char,char>> &color,
	char *filename)
{
	//get the number of triangles to read
	unsigned int numTri = getNumTriSTL(filename);
	//create host buffers to store data temporarily
	thrust::host_vector<thrust::tuple<float,float,float>> hp1(numTri);
	thrust::host_vector<thrust::tuple<float,float,float>> hp2(numTri);
	thrust::host_vector<thrust::tuple<float,float,float>> hp3(numTri);
	thrust::host_vector<thrust::tuple<char,char,char>> hcolor(numTri);
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
	//iterate
	for(; i < numTri; i++)
	{
		//read into buffers
		fread(norm, 4, 3, f);
		fread(v1, 4, 3, f);
		fread(v2, 4, 3, f);
		fread(v3, 4, 3, f);
		fread(&attr, 2, 1, f);
		//process buffers
		hp1[i] = thrust::make_tuple(v1[0], v1[1], v1[2]);
		hp2[i] = thrust::make_tuple(v2[0], v2[1], v2[2]);
		hp3[i] = thrust::make_tuple(v3[0], v3[1], v3[2]);
		hcolor[i] = getColor(norm);
	}

	//copy host vectors into device vectors
	//thrust::copy(p1.begin(), p1.end(), hp1.begin());
	p1 = hp1;
	p2 = hp2;
	p3 = hp3;
	color = hcolor;

	return i;
}

int main(int argc, char **argv)
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
}
