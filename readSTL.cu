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
	//close the file
	fclose(f);
	return 0;
}

int main(int argc, char **argv)
{
	readBinarySTL(argv[1]);
	return 0;
}
