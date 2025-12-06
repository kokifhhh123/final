#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <png.h>
#include <chrono>      // CPU timing
#include <cuda_runtime.h>  // CUDA timing

#include "imp.h"

// ------------ CPU timer helper ------------
double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// ------------ CUDA timer helper ------------
float cuda_time_ms(void (*func)(Image*, Image*, int, int),
                   Image* src, Image* dst,
                   int K, int max_iters)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    func(src, dst, K, max_iters);   // call CUDA version
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, stop, start);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char* argv[]) {
    // ./kmeans input.png output.png mode K [max_iters]
    if (argc != 5 && argc != 6) {
        std::cout << "Usage: ./kmeans input.png output.png "
                     "(seq|omp|cuda|cuda_opt) K [max_iters]\n";
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    const char* mode        = argv[3];
    int K = std::atoi(argv[4]);
    int max_iters = (argc == 6) ? std::atoi(argv[5]) : 20;

    unsigned char* buffer = nullptr;
    unsigned width, height, channels;

    if (read_png(input_path, &buffer, &height, &width, &channels) != 0) {
        std::cerr << "Failed to read input PNG.\n";
        return 1;
    }

    Image* src = createImage(width, height, channels);
    Image* dst = createImage(width, height, channels);

    std::memcpy(src->data, buffer, width * height * channels);
    std::memcpy(dst->data, buffer, width * height * channels);
    free(buffer);

    double t0, t1;
    float cuda_ms;

    if (std::strcmp(mode, "seq") == 0) {

        t0 = now();
        kmeans_seq(src, dst, K, max_iters);
        t1 = now();
        std::cout << "[SEQ] time = " << (t1 - t0) << " sec\n";

    }
    else if (std::strcmp(mode, "omp") == 0) {

        t0 = now();
        kmeans_omp(src, dst, K, max_iters);
        t1 = now();
        std::cout << "[OMP] time = " << (t1 - t0) << " sec\n";

    }
    else if (std::strcmp(mode, "cuda") == 0) {

        cuda_ms = cuda_time_ms(kmeans_cuda, src, dst, K, max_iters);
        std::cout << "[CUDA] time = " << cuda_ms / 1000.0f << " sec\n";

    }
    else if (std::strcmp(mode, "cuda_opt") == 0) {

        cuda_ms = cuda_time_ms(kmeans_cuda_opt, src, dst, K, max_iters);
        std::cout << "[CUDA_OPT] time = " << cuda_ms / 1000.0f << " sec\n";

    }
    else {
        std::cerr << "Invalid mode: " << mode
                  << " (expected seq|omp|cuda|cuda_opt)\n";
        freeImage(src);
        freeImage(dst);
        return 1;
    }

    write_png(output_path, dst->data, height, width, channels);

    freeImage(src);
    freeImage(dst);
    return 0;
}