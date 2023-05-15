#ifndef IMAGE_WRITER
#define IMAGE_WRITER

#include<cstdio>
#include<cstdlib>

typedef struct image
{
	char *data;
	int w, h;
}Image;

void initImage(Image *img, int w, int h);
void freeImage(Image *img);
void writeImage(Image *img, char *filename);

#endif
