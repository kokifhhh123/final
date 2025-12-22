#include <png.h>   // required for libpng functions

struct Image {
    unsigned char* data;
    unsigned width;
    unsigned height;
    unsigned channels;
};
struct Pixel { float r, g, b; float pad; };   
struct KMeansTiming {
    double io_host_prep_s = 0.0;   // host pixels + init centroids
    double h2d_init_s     = 0.0;   // initial H2D copies (pixels + centroids)
    double assign_s       = 0.0;   // assignment kernel only (sum over iters)
    double memset_s       = 0.0;   // cudaMemset (sum over iters)
    double accum_s        = 0.0;   // accumulate kernel only (sum over iters)
    double d2h_iter_s     = 0.0;   // per-iter D2H (accum + counts)
    double host_update_s  = 0.0;   // centroid update on CPU (sum over iters)
    double h2d_cent_s     = 0.0;   // per-iter H2D (centroids)
    double d2h_final_s    = 0.0;   // final D2H (assignments)
    double total_s        = 0.0;   // end-to-end inside this function
};

// 16 bytes

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels);
void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels);
Image* createImage(unsigned width, unsigned height, unsigned channels);
void freeImage(Image* img);
