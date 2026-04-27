#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
	//check filetype
	char *ftype = argv[1];
	for(; ftype[0] != '.' && ftype[0] != '\0'; ftype++);
	if(strcmp(ftype, ".stl") != 0)
		return -1; //not an .stl file
	//open the STL file
	FILE * f = fopen(argv[1], "r");
	//skip over the header
	fseek(f, 80, SEEK_SET);
	//get the data
	unsigned int numTri = 0;
	fread(&numTri, 4, 1, f);
	//print the result
	printf("%d\n", numTri);
	//close the file
	fclose(f);
}
