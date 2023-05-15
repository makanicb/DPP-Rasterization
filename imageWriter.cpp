#include<cstdio>
#include<cstdlib>
#include "imageWriter.h"

void initImage(Image *img, int w, int h)
{
	img->w = w;
	img->h = h;
	img->data = (char*)malloc(sizeof(char) * 3 * w * h);
	for(int i = 0; i < w * h; i++)
		img->data[i] = 0;
}

void freeImage(Image *img)
{
	free(img->data);
}

void writeImage(Image *img, char *filename)
{
	FILE* f = fopen(filename, "w");
	fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
	fwrite(img->data, sizeof(char), 3*img->w*img->h, f);
	fclose(f);
}
