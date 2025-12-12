#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cstring>
#include "heds/utils.h"
#include <png.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define MAX_K 128

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__        \
                      << " : " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__device__ float distance2_dev(const Pixel &a, const Pixel &b) {
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dr*dr + dg*dg + db*db;
}

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

        // int num_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            { int num_threads = omp_get_num_threads(); }

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







__global__ void assign_kernel(const Pixel* pixels,
                              const Pixel* centroids,
                              int* assignments,
                              int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Pixel p = pixels[idx];

    float best_dist = 1e30f;
    int best_k = 0;

    for (int k = 0; k < K; ++k) {
        float d = distance2_dev(p, centroids[k]);
        if (d < best_dist) {
            best_dist = d;
            best_k = k;
        }
    }
    assignments[idx] = best_k;
}

__global__ void accumulate_kernel(const Pixel* pixels,
                                  const int* assignments,
                                  Pixel* accum,
                                  int* counts,
                                  int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int k = assignments[idx];
    Pixel p = pixels[idx];

    atomicAdd(&accum[k].r, p.r);
    atomicAdd(&accum[k].g, p.g);
    atomicAdd(&accum[k].b, p.b);
    atomicAdd(&counts[k], 1);
}

void kmeans_cuda(Image* src, Image* dst, int K, int max_iters) {
    unsigned width  = src->width;
    unsigned height = src->height;
    unsigned channels = src->channels;

    size_t N = (size_t)width * (size_t)height;

    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels (RGB).\n";
        return;
    }

    // host pixels
    std::vector<Pixel> h_pixels(N);
    for (size_t i = 0; i < N; ++i) {
        h_pixels[i].r = src->data[i*channels + 0];
        h_pixels[i].g = src->data[i*channels + 1];
        h_pixels[i].b = src->data[i*channels + 2];
    }

    // host centroids
    std::srand(std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k) {
        size_t idx = std::rand() % N;
        h_centroids[k] = h_pixels[idx];
    }

    // device buffers
    Pixel* d_pixels     = nullptr;
    Pixel* d_centroids  = nullptr;
    Pixel* d_accum      = nullptr;
    int*   d_counts     = nullptr;
    int*   d_assignments = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(),
                          N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                          K * sizeof(Pixel), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);

    std::vector<Pixel> h_accum(K);
    std::vector<int>   h_counts(K);

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1) assignment
        assign_kernel<<<blocks, threads>>>(d_pixels, d_centroids,
                                           d_assignments, (int)N, K);
        CHECK_CUDA(cudaGetLastError());

        // 2) clear accumulators
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));

        // 3) accumulate
        accumulate_kernel<<<blocks, threads>>>(d_pixels, d_assignments,
                                               d_accum, d_counts, (int)N);
        CHECK_CUDA(cudaGetLastError());

        // 4) copy back accumulators
        CHECK_CUDA(cudaMemcpy(h_accum.data(), d_accum,
                              K * sizeof(Pixel), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts,
                              K * sizeof(int),   cudaMemcpyDeviceToHost));

        // 5) update centroids on host
        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > 0) {
                h_centroids[k].r = h_accum[k].r / h_counts[k];
                h_centroids[k].g = h_accum[k].g / h_counts[k];
                h_centroids[k].b = h_accum[k].b / h_counts[k];
            }
        }

        // 6) copy updated centroids to device
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                              K * sizeof(Pixel), cudaMemcpyHostToDevice));

        std::cout << "[CUDA] Iteration " << (iter+1)
                  << "/" << max_iters << " done.\n";
    }

    // 拿回 assignments
    std::vector<int> h_assignments(N);
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    // 寫回 dst
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i*channels + 0] = (unsigned char)h_centroids[k].r;
        dst->data[i*channels + 1] = (unsigned char)h_centroids[k].g;
        dst->data[i*channels + 2] = (unsigned char)h_centroids[k].b;
        if (channels == 4) {
            dst->data[i*channels + 3] = src->data[i*channels + 3];
        }
    }

    // free device memory
    CHECK_CUDA(cudaFree(d_pixels));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_assignments));
}






// Optimized assignment: centroids cached in shared memory
__global__ void assign_kernel_opt(const Pixel* pixels,
                                  const Pixel* centroids,
                                  int* assignments,
                                  int N, int K)
{
    extern __shared__ unsigned char smem[]; // dynamic shared
    Pixel* s_centroids = reinterpret_cast<Pixel*>(smem);

    // load centroids into shared memory
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s_centroids[k] = centroids[k];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Pixel p = pixels[idx];

    float best_dist = 1e30f;
    int best_k = 0;

    for (int k = 0; k < K; ++k) {
        float d = distance2_dev(p, s_centroids[k]);
        if (d < best_dist) {
            best_dist = d;
            best_k = k;
        }
    }
    assignments[idx] = best_k;
}

// Optimized accumulate: block-local reduction in shared memory
__global__ void accumulate_kernel_opt(const Pixel* pixels,
                                      const int* assignments,
                                      Pixel* g_accum,
                                      int* g_counts,
                                      int N, int K)
{
    extern __shared__ unsigned char smem[];
    Pixel* s_accum  = reinterpret_cast<Pixel*>(smem);
    int*   s_counts = reinterpret_cast<int*>(smem + K * sizeof(Pixel));

    // init block-local accumulators
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s_accum[k].r = 0.f;
        s_accum[k].g = 0.f;
        s_accum[k].b = 0.f;
        s_counts[k]  = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int cls = assignments[idx];
        Pixel p = pixels[idx];

        // still need atomic inside shared to avoid intra-block conflicts
        atomicAdd(&s_accum[cls].r, p.r);
        atomicAdd(&s_accum[cls].g, p.g);
        atomicAdd(&s_accum[cls].b, p.b);
        atomicAdd(&s_counts[cls], 1);
    }
    __syncthreads();

    // merge block-local accumulators into global once per cluster
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        if (s_counts[k] > 0) {
            atomicAdd(&g_accum[k].r, s_accum[k].r);
            atomicAdd(&g_accum[k].g, s_accum[k].g);
            atomicAdd(&g_accum[k].b, s_accum[k].b);
            atomicAdd(&g_counts[k], s_counts[k]);
        }
    }
}

void kmeans_cuda_opt(Image* src, Image* dst, int K, int max_iters) {
    if (K > MAX_K) {
        std::cerr << "K > MAX_K, please increase MAX_K.\n";
        return;
    }

    unsigned width  = src->width;
    unsigned height = src->height;
    unsigned channels = src->channels;

    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels (RGB).\n";
        return;
    }

    size_t N = (size_t)width * (size_t)height;

    // host pixels
    std::vector<Pixel> h_pixels(N);
    for (size_t i = 0; i < N; ++i) {
        h_pixels[i].r = src->data[i*channels + 0];
        h_pixels[i].g = src->data[i*channels + 1];
        h_pixels[i].b = src->data[i*channels + 2];
    }

    // host centroids init
    std::srand(std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k) {
        size_t idx = std::rand() % N;
        h_centroids[k] = h_pixels[idx];
    }

    // device buffers
    Pixel* d_pixels     = nullptr;
    Pixel* d_centroids  = nullptr;
    Pixel* d_accum      = nullptr;
    int*   d_counts     = nullptr;
    int*   d_assignments = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(),
                          N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                          K * sizeof(Pixel), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);

    std::vector<Pixel> h_accum(K);
    std::vector<int>   h_counts(K);

    // shared memory sizes
    size_t shmem_assign = K * sizeof(Pixel);
    size_t shmem_acc    = K * (sizeof(Pixel) + sizeof(int));

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1) assignment (centroids in shared)
        assign_kernel_opt<<<blocks, threads, shmem_assign>>>(
            d_pixels, d_centroids, d_assignments, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());

        // 2) clear global accumulators
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));

        // 3) block-local reduction in shared, then merge to global
        accumulate_kernel_opt<<<blocks, threads, shmem_acc>>>(
            d_pixels, d_assignments, d_accum, d_counts, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());

        // 4) copy back accumulators
        CHECK_CUDA(cudaMemcpy(h_accum.data(), d_accum,
                              K * sizeof(Pixel), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts,
                              K * sizeof(int), cudaMemcpyDeviceToHost));

        // 5) update centroids on host
        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > 0) {
                h_centroids[k].r = h_accum[k].r / h_counts[k];
                h_centroids[k].g = h_accum[k].g / h_counts[k];
                h_centroids[k].b = h_accum[k].b / h_counts[k];
            }
        }

        // 6) copy updated centroids to device
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                              K * sizeof(Pixel), cudaMemcpyHostToDevice));

        std::cout << "[CUDA_OPT] Iteration " << (iter+1)
                  << "/" << max_iters << " done.\n";
    }

    // 拿回 assignments
    std::vector<int> h_assignments(N);
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    // 寫回 dst
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i*channels + 0] = (unsigned char)h_centroids[k].r;
        dst->data[i*channels + 1] = (unsigned char)h_centroids[k].g;
        dst->data[i*channels + 2] = (unsigned char)h_centroids[k].b;
        if (channels == 4) {
            dst->data[i*channels + 3] = src->data[i*channels + 3];
        }
    }

    // free device memory
    CHECK_CUDA(cudaFree(d_pixels));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_assignments));
}







__global__ void accumulate_kernel_warp(const Pixel* pixels,
                                       const int* assignments,
                                       Pixel* g_accum,
                                       int* g_counts,
                                       int N)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane       = threadIdx.x & 31;            // lane id in warp
    unsigned full_mask = 0xffffffffu;

    // 判斷哪些 lane 有有效 pixel
    int  have_pixel = (global_idx < N);
    unsigned active = __ballot_sync(full_mask, have_pixel);

    if (!active) return;  // 這整個 warp 都沒事做

    // 每個 thread 自己的 k 與 (r,g,b,1)
    int   k      = -1;
    float r_val  = 0.0f;
    float g_val  = 0.0f;
    float b_val  = 0.0f;
    int   c_val  = 0;

    if (have_pixel) {
        k = assignments[global_idx];
        Pixel p = pixels[global_idx];
        r_val = p.r;
        g_val = p.g;
        b_val = p.b;
        c_val = 1;
    }

    // remaining 代表這個 warp 裡還沒被處理的 lanes
    unsigned remaining = active;

    // 反覆處理 warp 中「同一個 cluster」的那一群 lanes
    while (remaining) {
        // 找出剩下的其中一個 leader lane
        int leader_lane = __ffs(remaining) - 1;   // first set bit -> 0-based lane id

        // 把 leader 的 cluster id 廣播給整個 warp（只在 remaining 的 mask 裏）
        int leader_k = __shfl_sync(remaining, k, leader_lane);

        // 找出 warp 中，哪些 lanes 也是這個 cluster（且還在 remaining 裏）
        unsigned mask_k = __ballot_sync(remaining, have_pixel && (k == leader_k));

        // 只有屬於這個 cluster 的 lanes 會貢獻數值
        float r_sum = (mask_k & (1u << lane)) ? r_val : 0.0f;
        float g_sum = (mask_k & (1u << lane)) ? g_val : 0.0f;
        float b_sum = (mask_k & (1u << lane)) ? b_val : 0.0f;
        int   c_sum = (mask_k & (1u << lane)) ? c_val : 0;

        // 在該 mask_k 上做 warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            r_sum += __shfl_down_sync(mask_k, r_sum, offset);
            g_sum += __shfl_down_sync(mask_k, g_sum, offset);
            b_sum += __shfl_down_sync(mask_k, b_sum, offset);
            c_sum += __shfl_down_sync(mask_k, c_sum, offset);
        }

        // leader_lane 負責把整個 warp 對這個 cluster 的貢獻做一次 atomicAdd
        if (lane == leader_lane) {
            atomicAdd(&g_accum[leader_k].r, r_sum);
            atomicAdd(&g_accum[leader_k].g, g_sum);
            atomicAdd(&g_accum[leader_k].b, b_sum);
            atomicAdd(&g_counts[leader_k],  c_sum);
        }

        // 把已經處理過的 lanes 從 remaining 裡移除
        remaining &= ~mask_k;
    }
}

void kmeans_cuda_warp(Image* src, Image* dst, int K, int max_iters) {
    if (K > MAX_K) {
        std::cerr << "K > MAX_K, please increase MAX_K.\n";
        return;
    }

    unsigned width  = src->width;
    unsigned height = src->height;
    unsigned channels = src->channels;

    if (channels < 3) {
        std::cerr << "Image must have at least 3 channels (RGB).\n";
        return;
    }

    size_t N = (size_t)width * (size_t)height;

    // host pixels
    std::vector<Pixel> h_pixels(N);
    for (size_t i = 0; i < N; ++i) {
        h_pixels[i].r = src->data[i*channels + 0];
        h_pixels[i].g = src->data[i*channels + 1];
        h_pixels[i].b = src->data[i*channels + 2];
    }

    // host centroids init
    std::srand(std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k) {
        size_t idx = std::rand() % N;
        h_centroids[k] = h_pixels[idx];
    }

    // device buffers
    Pixel* d_pixels     = nullptr;
    Pixel* d_centroids  = nullptr;
    Pixel* d_accum      = nullptr;
    int*   d_counts     = nullptr;
    int*   d_assignments = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(),
                          N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                          K * sizeof(Pixel), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);

    std::vector<Pixel> h_accum(K);
    std::vector<int>   h_counts(K);

    // shared memory sizes
    size_t shmem_assign = K * sizeof(Pixel);
    // size_t shmem_acc    = K * (sizeof(Pixel) + sizeof(int));

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1) assignment (centroids in shared)
        assign_kernel_opt<<<blocks, threads, shmem_assign>>>(
            d_pixels, d_centroids, d_assignments, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());

        // 2) clear global accumulators
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));

        // 3) block-local reduction in shared, then merge to global
        accumulate_kernel_warp<<<blocks, threads>>>(
            d_pixels, d_assignments, d_accum, d_counts, (int)N
        );
        CHECK_CUDA(cudaGetLastError());

        // 4) copy back accumulators
        CHECK_CUDA(cudaMemcpy(h_accum.data(), d_accum,
                              K * sizeof(Pixel), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts,
                              K * sizeof(int), cudaMemcpyDeviceToHost));

        // 5) update centroids on host
        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > 0) {
                h_centroids[k].r = h_accum[k].r / h_counts[k];
                h_centroids[k].g = h_accum[k].g / h_counts[k];
                h_centroids[k].b = h_accum[k].b / h_counts[k];
            }
        }

        // 6) copy updated centroids to device
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                              K * sizeof(Pixel), cudaMemcpyHostToDevice));

        std::cout << "[CUDA_OPT] Iteration " << (iter+1)
                  << "/" << max_iters << " done.\n";
    }

    // 拿回 assignments
    std::vector<int> h_assignments(N);
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    // 寫回 dst
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i*channels + 0] = (unsigned char)h_centroids[k].r;
        dst->data[i*channels + 1] = (unsigned char)h_centroids[k].g;
        dst->data[i*channels + 2] = (unsigned char)h_centroids[k].b;
        if (channels == 4) {
            dst->data[i*channels + 3] = src->data[i*channels + 3];
        }
    }

    // free device memory
    CHECK_CUDA(cudaFree(d_pixels));
    CHECK_CUDA(cudaFree(d_centroids));
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_assignments));
}
