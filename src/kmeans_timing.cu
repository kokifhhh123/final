#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iostream>
// #include "timing.h"
#include "heds/algo.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)
#endif

#ifndef MAX_K
#define MAX_K 4096
#endif

// -------------------------- Device helpers --------------------------

__device__ __forceinline__ float distance2_dev(const Pixel& a, const Pixel& b) {
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dr * dr + dg * dg + db * db;
}


// -------------------------- Main timed API --------------------------

static inline double ms_to_s(float ms) { return (double)ms / 1000.0; }

KMeansTiming kmeans_cuda_warp_timed(const Image* src, Image* dst, int K, int max_iters)
{
    KMeansTiming tm;
    if (!src || !dst) return tm;
    if (K <= 0 || K > MAX_K) {
        std::cerr << "Invalid K.\n";
        return tm;
    }
    if (src->channels < 3 || dst->channels < 3) {
        std::cerr << "Image must have at least 3 channels.\n";
        return tm;
    }

    auto t_total0 = std::chrono::high_resolution_clock::now();

    unsigned width = src->width, height = src->height, channels = src->channels;
    size_t N = (size_t)width * (size_t)height;

    // ---------------- Host prep ----------------
    auto t_prep0 = std::chrono::high_resolution_clock::now();

    std::vector<Pixel> h_pixels(N);
    for (size_t i = 0; i < N; ++i) {
        h_pixels[i].r = (float)src->data[i * channels + 0];
        h_pixels[i].g = (float)src->data[i * channels + 1];
        h_pixels[i].b = (float)src->data[i * channels + 2];
    }

    std::srand((unsigned)std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k) {
        h_centroids[k] = h_pixels[(size_t)std::rand() % N];
    }

    std::vector<Pixel> h_accum(K);
    std::vector<int>   h_counts(K);
    std::vector<int>   h_assignments(N);

    auto t_prep1 = std::chrono::high_resolution_clock::now();
    tm.io_host_prep_s = std::chrono::duration<double>(t_prep1 - t_prep0).count();

    // ---------------- Device buffers ----------------
    Pixel *d_pixels=nullptr, *d_centroids=nullptr, *d_accum=nullptr;
    int   *d_counts=nullptr, *d_assignments=nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);
    size_t shmem_assign = (size_t)K * sizeof(Pixel);

    // ---------------- Events (reused) ----------------
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    float ms = 0.f;

    // ---------------- Initial H2D ----------------
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    tm.h2d_init_s += ms_to_s(ms);

    // ---------------- Iterations ----------------
    for (int iter = 0; iter < max_iters; ++iter) {

        // (A) Assignment kernel time
        CHECK_CUDA(cudaEventRecord(e0));
        assign_kernel_opt<<<blocks, threads, shmem_assign>>>(
            d_pixels, d_centroids, d_assignments, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.assign_s += ms_to_s(ms);

        // (B) Memset time
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.memset_s += ms_to_s(ms);

        // (C) Accumulate kernel time
        CHECK_CUDA(cudaEventRecord(e0));
        accumulate_kernel_warp<<<blocks, threads>>>(
            d_pixels, d_assignments, d_accum, d_counts, (int)N
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.accum_s += ms_to_s(ms);

        // (D) D2H for accum + counts
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemcpy(h_accum.data(),  d_accum,  K * sizeof(Pixel), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, K * sizeof(int),   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.d2h_iter_s += ms_to_s(ms);

        // (E) Host centroid update time (CPU)
        auto t_up0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > 0) {
                float inv = 1.0f / (float)h_counts[k];
                h_centroids[k].r = h_accum[k].r * inv;
                h_centroids[k].g = h_accum[k].g * inv;
                h_centroids[k].b = h_accum[k].b * inv;
            }
        }
        auto t_up1 = std::chrono::high_resolution_clock::now();
        tm.host_update_s += std::chrono::duration<double>(t_up1 - t_up0).count();

        // (F) H2D updated centroids
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), K * sizeof(Pixel), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.h2d_cent_s += ms_to_s(ms);
    }

    // ---------------- Final D2H assignments ----------------
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    tm.d2h_final_s += ms_to_s(ms);

    // ---------------- Write result into dst (CPU) ----------------
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i * channels + 0] = (unsigned char)h_centroids[k].r;
        dst->data[i * channels + 1] = (unsigned char)h_centroids[k].g;
        dst->data[i * channels + 2] = (unsigned char)h_centroids[k].b;
        if (channels == 4) dst->data[i * channels + 3] = src->data[i * channels + 3];
    }

    // ---------------- Cleanup ----------------
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaFree(d_pixels));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_assignments));

    auto t_total1 = std::chrono::high_resolution_clock::now();
    tm.total_s = std::chrono::duration<double>(t_total1 - t_total0).count();
    return tm;
}




KMeansTiming kmeans_cuda_opt_timed(const Image* src, Image* dst, int K, int max_iters)
{
    KMeansTiming tm;
    if (!src || !dst) return tm;
    if (K <= 0 || K > MAX_K) {
        std::cerr << "Invalid K.\n";
        return tm;
    }
    if (src->channels < 3 || dst->channels < 3) {
        std::cerr << "Image must have at least 3 channels.\n";
        return tm;
    }

    auto t_total0 = std::chrono::high_resolution_clock::now();

    unsigned width = src->width, height = src->height, channels = src->channels;
    size_t N = (size_t)width * (size_t)height;

    // ---------------- Host prep ----------------
    auto t_prep0 = std::chrono::high_resolution_clock::now();

    std::vector<Pixel> h_pixels(N);
    for (size_t i = 0; i < N; ++i) {
        h_pixels[i].r = src->data[i * channels + 0];
        h_pixels[i].g = src->data[i * channels + 1];
        h_pixels[i].b = src->data[i * channels + 2];
    }

    std::srand((unsigned)std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k)
        h_centroids[k] = h_pixels[(size_t)std::rand() % N];

    std::vector<Pixel> h_accum(K);
    std::vector<int>   h_counts(K);
    std::vector<int>   h_assignments(N);

    auto t_prep1 = std::chrono::high_resolution_clock::now();
    tm.io_host_prep_s =
        std::chrono::duration<double>(t_prep1 - t_prep0).count();

    // ---------------- Device alloc ----------------
    Pixel *d_pixels=nullptr, *d_centroids=nullptr, *d_accum=nullptr;
    int   *d_counts=nullptr, *d_assignments=nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);
    size_t shmem_assign = (size_t)K * sizeof(Pixel);
    size_t shmem_acc    = (size_t)K * (sizeof(Pixel) + sizeof(int));

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    float ms = 0.f;

    // ---------------- H2D init ----------------
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(),
                          N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                          K * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    tm.h2d_init_s += ms / 1000.0;

    // ---------------- Iterations ----------------
    for (int iter = 0; iter < max_iters; ++iter) {

        // assignment
        CHECK_CUDA(cudaEventRecord(e0));
        assign_kernel_opt<<<blocks, threads, shmem_assign>>>(
            d_pixels, d_centroids, d_assignments, (int)N, K);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.assign_s += ms / 1000.0;

        // memset
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.memset_s += ms / 1000.0;

        // accumulate (shared-memory opt)
        CHECK_CUDA(cudaEventRecord(e0));
        accumulate_kernel_opt<<<blocks, threads, shmem_acc>>>(
            d_pixels, d_assignments, d_accum, d_counts, (int)N, K);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.accum_s += ms / 1000.0;

        // D2H per-iter
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemcpy(h_accum.data(),  d_accum,
                              K * sizeof(Pixel), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts,
                              K * sizeof(int),   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.d2h_iter_s += ms / 1000.0;

        // host update
        auto t_up0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > 0) {
                h_centroids[k].r = h_accum[k].r / (float)h_counts[k];
                h_centroids[k].g = h_accum[k].g / (float)h_counts[k];
                h_centroids[k].b = h_accum[k].b / (float)h_counts[k];
            }
        }
        auto t_up1 = std::chrono::high_resolution_clock::now();
        tm.host_update_s +=
            std::chrono::duration<double>(t_up1 - t_up0).count();

        // H2D centroids
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                              K * sizeof(Pixel), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        tm.h2d_cent_s += ms / 1000.0;
    }

    // ---------------- Final D2H ----------------
    CHECK_CUDA(cudaEventRecord(e0));
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    tm.d2h_final_s += ms / 1000.0;

    // write result
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i * channels + 0] = (unsigned char)h_centroids[k].r;
        dst->data[i * channels + 1] = (unsigned char)h_centroids[k].g;
        dst->data[i * channels + 2] = (unsigned char)h_centroids[k].b;
        if (channels == 4)
            dst->data[i * channels + 3] = src->data[i * channels + 3];
    }

    // cleanup
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaFree(d_pixels));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_assignments));

    auto t_total1 = std::chrono::high_resolution_clock::now();
    tm.total_s =
        std::chrono::duration<double>(t_total1 - t_total0).count();

    return tm;
}