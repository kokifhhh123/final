#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include "utils.h"

//----------------------------------------------------
// Pixel struct and distance
//----------------------------------------------------
struct Pixel {
    float r, g, b;
};

float distance2(const Pixel &a, const Pixel &b) {
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dr*dr + dg*dg + db*db;
}

//----------------------------------------------------
// Main: K-means color quantization
//----------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./kmeans_seq_png input.png output.png K\n";
        return 0;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int K = std::atoi(argv[3]);

    unsigned width, height, channels;
    unsigned char* img = nullptr;

    if (read_png(input_path, &img, &height, &width, &channels) != 0) {
        std::cout << "Failed to load PNG.\n";
        return 0;
    }

    if (channels < 3) {
        std::cout << "Image must be RGB or RGBA.\n";
        return 0;
    }

    size_t N = width * height;
    std::vector<Pixel> pixels(N);

    // convert to Pixel struct (ignore alpha channel if exists)
    for (size_t i = 0; i < N; i++) {
        pixels[i].r = img[i*channels + 0];
        pixels[i].g = img[i*channels + 1];
        pixels[i].b = img[i*channels + 2];
    }

    // initialize centroids randomly
    std::srand(std::time(nullptr));
    std::vector<Pixel> centroids(K);
    for (int k = 0; k < K; k++) {
        int idx = std::rand() % N;
        centroids[k] = pixels[idx];
    }

    std::vector<int> assignments(N, 0);
    std::vector<Pixel> new_centroids(K);
    std::vector<int> counts(K);

    int max_iters = 20;

    for (int iter = 0; iter < max_iters; ++iter) {
        // assign each pixel to nearest centroid
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

        // reset accumulators
        for (int k = 0; k < K; k++) {
            new_centroids[k] = {0,0,0};
            counts[k] = 0;
        }

        // update centroids
        for (size_t i = 0; i < N; i++) {
            int k = assignments[i];
            new_centroids[k].r += pixels[i].r;
            new_centroids[k].g += pixels[i].g;
            new_centroids[k].b += pixels[i].b;
            counts[k]++;
        }

        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                centroids[k].r = new_centroids[k].r / counts[k];
                centroids[k].g = new_centroids[k].g / counts[k];
                centroids[k].b = new_centroids[k].b / counts[k];
            }
        }

        std::cout << "Iteration " << iter + 1 << "/" << max_iters << " done.\n";
    }

    // write quantized image
    for (size_t i = 0; i < N; i++) {
        int k = assignments[i];
        img[i*channels + 0] = (unsigned char)centroids[k].r;
        img[i*channels + 1] = (unsigned char)centroids[k].g;
        img[i*channels + 2] = (unsigned char)centroids[k].b;
    }

    write_png(output_path, img, height, width, channels);

    free(img);

    std::cout << "Saved output: " << output_path << "\n";
    return 0;
}
