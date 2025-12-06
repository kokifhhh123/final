#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cstring>
#include "utils.h"
#include <png.h>
#include <omp.h>

float distance2(const Pixel &a, const Pixel &b) {
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dr*dr + dg*dg + db*db;
}

void kmeans_seq(Image* src, Image* dst, int K, int max_iters) {
    unsigned width = src->width;
    unsigned height = src->height;
    unsigned channels = src->channels;

    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels (RGB).\n";
        return;
    }

    size_t N = width * height;
    std::vector<Pixel> pixels(N);

    // load pixels from src (ignore alpha if present)
    for (size_t i = 0; i < N; i++) {
        pixels[i].r = src->data[i*channels + 0];
        pixels[i].g = src->data[i*channels + 1];
        pixels[i].b = src->data[i*channels + 2];
    }

    std::srand(std::time(nullptr));
    std::vector<Pixel> centroids(K);
    for (int k = 0; k < K; k++) {
        size_t idx = std::rand() % N;
        centroids[k] = pixels[idx];
    }

    std::vector<int> assignments(N, 0);
    std::vector<Pixel> new_centroids(K);
    std::vector<int> counts(K);

    for (int iter = 0; iter < max_iters; ++iter) {
        // assignment
        for (size_t i = 0; i < N; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int best_k = 0;
            for (int k = 0; k < K; k++) {
                float d = distance2(pixels[i], centroids[k]);
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        // reset centroids
        for (int k = 0; k < K; k++) {
            new_centroids[k] = {0,0,0};
            counts[k] = 0;
        }

        // accumulate
        for (size_t i = 0; i < N; i++) {
            int k = assignments[i];
            new_centroids[k].r += pixels[i].r;
            new_centroids[k].g += pixels[i].g;
            new_centroids[k].b += pixels[i].b;
            counts[k]++;
        }
        // update
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                centroids[k].r = new_centroids[k].r / counts[k];
                centroids[k].g = new_centroids[k].g / counts[k];
                centroids[k].b = new_centroids[k].b / counts[k];
            }
        }
        std::cout << "Iteration " << (iter+1) << "/" << max_iters << " done.\n";
    }
    // write result to dst
    for (size_t i = 0; i < N; i++) {
        int k = assignments[i];
        dst->data[i*channels + 0] = (unsigned char)centroids[k].r;
        dst->data[i*channels + 1] = (unsigned char)centroids[k].g;
        dst->data[i*channels + 2] = (unsigned char)centroids[k].b;
        // alpha channel (if exists) copied from src
        if (channels == 4) {
            dst->data[i*channels + 3] = src->data[i*channels + 3];
        }
    }
}

void kmeans_omp(Image* src, Image* dst, int K, int max_iters) {
    unsigned width  = src->width;
    unsigned height = src->height;
    unsigned channels = src->channels;

    size_t N = width * height;

    std::vector<Pixel> pixels(N);
    for (size_t i = 0; i < N; i++) {
        pixels[i].r = src->data[i*channels + 0];
        pixels[i].g = src->data[i*channels + 1];
        pixels[i].b = src->data[i*channels + 2];
    }

    // Initialize centroids
    std::srand(std::time(nullptr));
    std::vector<Pixel> centroids(K);
    for (int k = 0; k < K; k++) {
        centroids[k] = pixels[std::rand() % N];
    }

    std::vector<int> assignments(N, 0);

    for (int iter = 0; iter < max_iters; iter++) {
        // ========================
        // 1) Assignment step
        // ========================
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int best_k = 0;

            for (int k = 0; k < K; k++) {
                float d = distance2(pixels[i], centroids[k]);
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        // ========================
        // 2) Update centroids with thread-local reduction
        // ========================
        std::vector<Pixel> new_centroids(K, {0,0,0});
        std::vector<int> counts(K, 0);

        int num_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            { num_threads = omp_get_num_threads(); }

            std::vector<Pixel> local_sum(K, {0,0,0});
            std::vector<int>   local_count(K, 0);

            #pragma omp for nowait
            for (size_t i = 0; i < N; i++) {
                int k = assignments[i];
                local_sum[k].r += pixels[i].r;
                local_sum[k].g += pixels[i].g;
                local_sum[k].b += pixels[i].b;
                local_count[k]++;
            }

            // merge thread-local results
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    new_centroids[k].r += local_sum[k].r;
                    new_centroids[k].g += local_sum[k].g;
                    new_centroids[k].b += local_sum[k].b;
                    counts[k] += local_count[k];
                }
            }
        }

        // Final centroid update
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                centroids[k].r = new_centroids[k].r / counts[k];
                centroids[k].g = new_centroids[k].g / counts[k];
                centroids[k].b = new_centroids[k].b / counts[k];
            }
        }

        std::cout << "[OMP] Iteration " << iter+1 << "/" << max_iters << " done.\n";
    }

    // ========================
    // Write result into dst
    // ========================
    for (size_t i = 0; i < N; i++) {
        int k = assignments[i];
        dst->data[i*channels + 0] = (unsigned char)centroids[k].r;
        dst->data[i*channels + 1] = (unsigned char)centroids[k].g;
        dst->data[i*channels + 2] = (unsigned char)centroids[k].b;

        if (channels == 4)
            dst->data[i*channels + 3] = src->data[i*channels + 3];
    }
}

void kmeans_cuda(Image* src, Image* dst, int K, int max_iters) {
    // TODO: implement CUDA naive version
    std::cout << "[WARN] kmeans_cuda not implemented yet, fallback to seq.\n";
    kmeans_seq(src, dst, K, max_iters);
}

void kmeans_cuda_opt(Image* src, Image* dst, int K, int max_iters) {
    // todo: implement CUDA optimized version
    std::cout << "[WARN] kmeans_cuda_opt not implemented yet, fallback to seq.\n";
    kmeans_seq(src, dst, K, max_iters);
}