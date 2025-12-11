#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <png.h>
#include <chrono>      // CPU timing
#include <cuda_runtime.h>  // CUDA timing
#include "heds/algo.h"

// ------------ CPU timer helper ------------
double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
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
    const char* algo        = argv[3];
    const char* mode        = argv[4];
    int K = std::atoi(argv[5]);
    int max_iters = (argc == 7) ? std::atoi(argv[6]) : 20;

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


    if (std::strcmp(algo, "kmeans") == 0) {
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
        else if (strcmp(mode, "cuda") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kmeans_cuda(src, dst, K, max_iters);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cuda_ms, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            std::cout << "[CUDA] time = " << cuda_ms / 1000.0f << " sec\n";
        }
        else if (std::strcmp(mode, "cuda_opt") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kmeans_cuda_opt(src, dst, K, max_iters);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[CUDA_OPT] time = " << ms / 1000.0f << " sec\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        else if (std::strcmp(mode, "cuda_opt_more") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kmeans_cuda_opt_more(src, dst, K, max_iters);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[CUDA_OPT] time = " << ms / 1000.0f << " sec\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        else if (std::strcmp(mode, "cuda_opt_warp") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kmeans_cuda_warp(src, dst, K, max_iters);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[CUDA_OPT] time = " << ms / 1000.0f << " sec\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        // else if (std::strcmp(mode, "cuda_opt_soa") == 0) {
        //     cudaEvent_t start, stop;
        //     cudaEventCreate(&start);
        //     cudaEventCreate(&stop);

        //     cudaEventRecord(start);
        //     kmeans_cuda_opt_more_soa(src, dst, K, max_iters);
        //     cudaEventRecord(stop);
        //     cudaEventSynchronize(stop);

        //     float ms = 0.0f;
        //     cudaEventElapsedTime(&ms, start, stop);
        //     std::cout << "[CUDA_OPT] time = " << ms / 1000.0f << " sec\n";

        //     cudaEventDestroy(start);
        //     cudaEventDestroy(stop);
        // }
        else {
            std::cerr << "Invalid mode: " << mode
                    << " (expected seq|omp|cuda|cuda_opt)\n";
            freeImage(src);
            freeImage(dst);
            return 1;
        }
    } 
    else if (strcmp(algo, "slic") == 0) {
        if (strcmp(mode, "seq") == 0) {
            t0 = now();
            slic_seq(src, dst, K, max_iters);
            t1 = now();
            std::cout << "[SLIC-SEQ] time = " << (t1 - t0) << " sec\n";
        }
        else if (std::strcmp(mode, "omp") == 0) {
            
            t0 = now();
            std::cout << "hello" << std::endl;
            slic_omp(src, dst, K, max_iters);
            t1 = now();
            std::cout << "[slic-OMP] time = " << (t1 - t0) << " sec\n";
        }
        else if (std::strcmp(mode, "cuda") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            slic_cuda(src, dst, K, max_iters);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[slic-CUDA] time = " << ms / 1000.0f << " sec\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        else if (std::strcmp(mode, "cudaopt") == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            slic_cuda_opt(src, dst, K, max_iters);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "[slic-CUDA_OPT] time = " << ms / 1000.0f << " sec\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    
    write_png(output_path, dst->data, height, width, channels);
    freeImage(src);
    freeImage(dst);
    return 0;
}