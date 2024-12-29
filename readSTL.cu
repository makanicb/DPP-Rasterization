#include <cstdlib>
#include <cstdio>
#include <iostream>
/*
    A function for reading a binary STL file to standard output
    Parameters:
    - filename: a pointer to an array of characters containing the
      name of the file to read
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

int main(int argc, char **argv)
{
	readBinarySTL(argv[1]);
	return 0;
}
