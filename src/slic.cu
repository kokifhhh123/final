#include "heds/algo.h"

#include <cmath>
#include <vector>
// #include <limits>
#include <iostream>

// ---------------------------------------------------------
// 工具：RGB → Lab （簡易版本）
// ---------------------------------------------------------
static void rgb2lab(unsigned char R, unsigned char G, unsigned char B,
                    float &L, float &a, float &b)
{
    // normalize
    float r = R / 255.0f;
    float g = G / 255.0f;
    float bl = B / 255.0f;

    // sRGB to XYZ
    auto f = [](float x) {
        return (x > 0.04045f) ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92f);
    };
    r = f(r); g = f(g); bl = f(bl);

    float X = r * 0.4124564f + g * 0.3575761f + bl * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + bl * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + bl * 0.9503041f;

    // normalize for D65 white point
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

// ---------------------------------------------------------
// SLIC Sequential Version
// ---------------------------------------------------------
void slic_seq(Image* src, Image* dst, int K, int max_iters)
{
    unsigned W = src->width;
    unsigned H = src->height;
    unsigned C = src->channels;
    size_t N = (size_t)W * H;

    if (C < 3) {
        std::cerr << "SLIC requires RGB image.\n";
        return;
    }

    // -----------------------------------------------------
    // Step 1: Precompute Lab + XY feature
    // feature vector: [L, a, b, x, y]
    // -----------------------------------------------------
    std::vector<float> L(N), A(N), B(N), X(N), Y(N);

    for (unsigned y = 0; y < H; ++y) {
        for (unsigned x = 0; x < W; ++x) {
            size_t idx = y * W + x;

            unsigned char r = src->data[idx*C + 0];
            unsigned char g = src->data[idx*C + 1];
            unsigned char b = src->data[idx*C + 2];

            rgb2lab(r, g, b, L[idx], A[idx], B[idx]);
            X[idx] = (float)x;
            Y[idx] = (float)y;
        }
    }

    // -----------------------------------------------------
    // Step 2: Initialize cluster centers on regular grid
    // -----------------------------------------------------
    float S = sqrtf((float)N / K);   // grid interval
    std::vector<float> cL, cA, cB, cX, cY;
    std::vector<int> counts;
    cL.reserve(K); cA.reserve(K); cB.reserve(K); 
    cX.reserve(K); cY.reserve(K);

    for (float cy = S/2; cy < H; cy += S) {
        for (float cx = S/2; cx < W; cx += S) {
            size_t ix = (int)cx;
            size_t iy = (int)cy;
            size_t idx = iy * W + ix;
            cL.push_back(L[idx]);
            cA.push_back(A[idx]);
            cB.push_back(B[idx]);
            cX.push_back(cx);
            cY.push_back(cy);
        }
    }

    int Kactual = cL.size();
    counts.resize(Kactual);

    // Label array
    std::vector<int> labels(N, -1);

    // -----------------------------------------------------
    // Step 3: Iterate SLIC loops
    // -----------------------------------------------------
    const float m = 10.0f;   // compactness parameter (可調整)

    for (int iter = 0; iter < max_iters; ++iter) {

        // reset
        std::fill(counts.begin(), counts.end(), 0);

        std::vector<float> sumL(Kactual, 0);
        std::vector<float> sumA(Kactual, 0);
        std::vector<float> sumB(Kactual, 0);
        std::vector<float> sumX(Kactual, 0);
        std::vector<float> sumY(Kactual, 0);

        // ---------------------------------------------
        // Assign step: each cluster searches local 2S × 2S region
        // ---------------------------------------------
        for (int k = 0; k < Kactual; ++k) {
            int cx = (int)cX[k];
            int cy = (int)cY[k];

            int x0 = std::max(0,   cx - (int)S);
            int x1 = std::min((int)W-1, cx + (int)S);
            int y0 = std::max(0,   cy - (int)S);
            int y1 = std::min((int)H-1, cy + (int)S);

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    size_t idx = y*W + x;

                    float dL = L[idx] - cL[k];
                    float dA = A[idx] - cA[k];
                    float dB = B[idx] - cB[k];
                    float dcolor = dL*dL + dA*dA + dB*dB;

                    float dx = x - cX[k];
                    float dy = y - cY[k];
                    float dspace = dx*dx + dy*dy;

                    float D = dcolor + (dspace / (S*S)) * (m*m);

                    if (labels[idx] < 0 || D < labels[idx]) {
                        labels[idx] = k;
                    }
                }
            }
        }

        // ---------------------------------------------
        // Update step: recompute cluster centers
        // ---------------------------------------------
        for (int k = 0; k < Kactual; ++k) {
            sumL[k] = sumA[k] = sumB[k] = 0;
            sumX[k] = sumY[k] = 0;
            counts[k] = 0;
        }

        for (size_t i = 0; i < N; ++i) {
            int k = labels[i];
            sumL[k] += L[i];
            sumA[k] += A[i];
            sumB[k] += B[i];
            sumX[k] += X[i];
            sumY[k] += Y[i];
            counts[k]++;
        }

        for (int k = 0; k < Kactual; ++k) {
            if (counts[k] > 0) {
                cL[k] = sumL[k] / counts[k];
                cA[k] = sumA[k] / counts[k];
                cB[k] = sumB[k] / counts[k];
                cX[k] = sumX[k] / counts[k];
                cY[k] = sumY[k] / counts[k];
            }
        }

        std::cout << "[SLIC-SEQ] Iter " << iter+1 << "/" << max_iters << " done.\n";
    }

    // -----------------------------------------------------
    // Step 4: Write output using average RGB per superpixel
    // -----------------------------------------------------
    std::vector<float> sumR(Kactual, 0);
    std::vector<float> sumG(Kactual, 0);
    std::vector<float> sumB(Kactual, 0);
    std::vector<int> countPix(Kactual, 0);

    // accumulate original RGB for each superpixel
    for (size_t i = 0; i < N; ++i) {
        int k = labels[i];
        sumR[k] += src->data[i*C + 0];
        sumG[k] += src->data[i*C + 1];
        sumB[k] += src->data[i*C + 2];
        countPix[k]++;
    }

    // finalize average RGB
    std::vector<unsigned char> avgR(Kactual), avgG(Kactual), avgB(Kactual);
    for (int k = 0; k < Kactual; ++k) {
        if (countPix[k] > 0) {
            avgR[k] = (unsigned char)(sumR[k] / countPix[k]);
            avgG[k] = (unsigned char)(sumG[k] / countPix[k]);
            avgB[k] = (unsigned char)(sumB[k] / countPix[k]);
        } else {
            avgR[k] = avgG[k] = avgB[k] = 0;
        }
    }

    // assign color to output
    for (size_t i = 0; i < N; ++i) {
        int k = labels[i];
        dst->data[i*C + 0] = avgR[k];
        dst->data[i*C + 1] = avgG[k];
        dst->data[i*C + 2] = avgB[k];
        if (C == 4) dst->data[i*C + 3] = 255;
    }
}
