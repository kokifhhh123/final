#include <ctime>
#include <cstring>
#include "utils.h"
#include <png.h>

void kmeans_seq(Image* src, Image* dst, int K, int max_iters);
void kmeans_omp(Image* src, Image* dst, int K, int max_iters);
void kmeans_cuda(Image* src, Image* dst, int K, int max_iters);
void kmeans_cuda_opt(Image* src, Image* dst, int K, int max_iters);
void kmeans_cuda_opt_more(Image* src, Image* dst, int K, int max_iters);
void kmeans_cuda_warp(Image* src, Image* dst, int K, int max_iters);
// void kmeans_cuda_opt_more_soa(Image* src, Image* dst, int K, int max_iters);

void slic_seq(Image* src, Image* dst, int K, int max_iters);
void slic_omp(Image* src, Image* dst, int K, int max_iters);
void slic_cuda(Image* src, Image* dst, int K, int max_iters);
// void slic_cuda_opt(Image* src, Image* dst, int K, int max_iters);


__global__ void k_update_centers(float* cL, float* cA, float* cB,
                                 float* cX, float* cY,
                                 const float* accL, const float* accA, const float* accB,
                                 const float* accX, const float* accY, const int* accCount,
                                 int Kactual);

__global__ void k_reconstruct(const unsigned char* __restrict__ src,
                              unsigned char* dst,
                              const int* __restrict__ labels,
                              float* sumR, float* sumG, float* sumB, int* count,
                              int W, int H, int C, int Kactual);

__device__ void rgb2lab_gpu(unsigned char R, unsigned char G, unsigned char B,
                            float &L, float &a, float &b);

__global__ void k_write_pixels(unsigned char* dst,
                               const int* __restrict__ labels,
                               const float* sumR, const float* sumG, const float* sumB, const int* count,
                               int W, int H, int C, int Kactual);

__global__ void k_rgb2lab(const unsigned char* __restrict__ src,
                          float* d_L, float* d_A, float* d_B,
                          int W, int H, int C);
// rgb2lab 函式保持不變，直接沿用即可
static void rgb2lab(unsigned char R, unsigned char G, unsigned char B,
                    float &L, float &a, float &b)
{
    float r = R / 255.0f;
    float g = G / 255.0f;
    float bl = B / 255.0f;

    auto f = [](float x) {
        return (x > 0.04045f) ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92f);
    };
    r = f(r); g = f(g); bl = f(bl);

    float X = r * 0.4124564f + g * 0.3575761f + bl * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + bl * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + bl * 0.9503041f;

    X /= 0.95047f;  
    Z /= 1.08883f;

    auto f2 = [](float t) {
        return (t > 0.008856f) ? powf(t, 1.f/3.f) : (7.787f*t + 16.f/116.f);
    };
    float fx = f2(X);
    float fy = f2(Y);
    float fz = f2(Z);

    L = 116.f * fy - 16.f;
    a = 500.f * (fx - fy);
    b = 200.f * (fy - fz);
}

