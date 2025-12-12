
void kmeans_cuda_opt_more(Image* src, Image* dst, int K, int max_iters) {
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

    // initial centroids on host
    std::srand(std::time(nullptr));
    std::vector<Pixel> h_centroids(K);
    for (int k = 0; k < K; ++k) {
        size_t idx = std::rand() % N;
        h_centroids[k] = h_pixels[idx];
    }

    // device buffers
    Pixel* d_pixels      = nullptr;
    Pixel* d_centroids   = nullptr;
    Pixel* d_accum       = nullptr;
    int*   d_counts      = nullptr;
    int*   d_assignments = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pixels,      N * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_centroids,   K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_accum,       K * sizeof(Pixel)));
    CHECK_CUDA(cudaMalloc(&d_counts,      K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_assignments, N * sizeof(int)));

    // copy pixels and initial centroids to device once
    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(),
                          N * sizeof(Pixel), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(),
                          K * sizeof(Pixel), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((N + threads - 1) / threads);

    // shared memory sizes for opt kernels
    size_t shmem_assign = K * sizeof(Pixel);
    size_t shmem_acc    = K * (sizeof(Pixel) + sizeof(int));

    // kernel config for centroid update
    int threadsK = 128;
    int blocksK  = (K + threadsK - 1) / threadsK;

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1) assignment with centroids cached in shared memory
        assign_kernel_opt<<<blocks, threads, shmem_assign>>>(
            d_pixels, d_centroids, d_assignments, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());

        // 2) clear accumulators on device
        CHECK_CUDA(cudaMemset(d_accum,  0, K * sizeof(Pixel)));
        CHECK_CUDA(cudaMemset(d_counts, 0, K * sizeof(int)));

        // 3) block-local reduction in shared, then merge to global
        accumulate_kernel_opt<<<blocks, threads, shmem_acc>>>(
            d_pixels, d_assignments, d_accum, d_counts, (int)N, K
        );
        CHECK_CUDA(cudaGetLastError());

        // 4) update centroids on GPU (no D2H / H2D in the loop)
        update_centroids_kernel<<<blocksK, threadsK>>>(
            d_centroids, d_accum, d_counts, K
        );
        CHECK_CUDA(cudaGetLastError());

        std::cout << "[CUDA_OPT_MORE] Iteration " << (iter+1)
                  << "/" << max_iters << " done.\n";
    }

    // copy final centroids and assignments back to host once
    std::vector<Pixel> h_centroids_final(K);
    std::vector<int>   h_assignments(N);

    CHECK_CUDA(cudaMemcpy(h_centroids_final.data(), d_centroids,
                          K * sizeof(Pixel), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_assignments.data(), d_assignments,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    // write result to dst
    for (size_t i = 0; i < N; ++i) {
        int k = h_assignments[i];
        dst->data[i*channels + 0] = (unsigned char)h_centroids_final[k].r;
        dst->data[i*channels + 1] = (unsigned char)h_centroids_final[k].g;
        dst->data[i*channels + 2] = (unsigned char)h_centroids_final[k].b;
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
