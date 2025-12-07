#include <png.h>   // required for libpng functions

struct Image {
    unsigned char* data;
    unsigned width;
    unsigned height;
    unsigned channels;
};

struct Pixel { float r, g, b; float pad; };   // 16 bytes

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels);
void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels);
Image* createImage(unsigned width, unsigned height, unsigned channels);
void freeImage(Image* img);