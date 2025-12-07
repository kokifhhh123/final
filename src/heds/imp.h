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
void kmeans_cuda_opt_more_soa(Image* src, Image* dst, int K, int max_iters);
