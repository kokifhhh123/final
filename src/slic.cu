#include "heds/algo.h"

#include <cmath>
#include <vector>
// #include <limits>
#include <iostream>
#include <cfloat>
#include <omp.h>



// ---------------------------------------------------------
// SLIC Sequential Version (Pixel-Centric Logic)
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
    // Step 1: Precompute Lab + XY
    // -----------------------------------------------------
    std::vector<float> L(N), A(N), B(N);
    // 這裡我們不需要存 X, Y 陣列，可以直接在迴圈計算 (i % W, i / W) 省記憶體
    
    for (size_t i = 0; i < N; ++i) {
        unsigned char r = src->data[i*C + 0];
        unsigned char g = src->data[i*C + 1];
        unsigned char b = src->data[i*C + 2];
        rgb2lab(r, g, b, L[i], A[i], B[i]);
    }

    // -----------------------------------------------------
    // Step 2: Initialize cluster centers & Grid Map
    // -----------------------------------------------------
    float S = sqrtf((float)N / K);
    
    // 建立 Grid Map 機制 (對應 CUDA 版邏輯)
    int grid_w = (int)ceil(W / S);
    int grid_h = (int)ceil(H / S);
    std::vector<int> grid_to_k(grid_w * grid_h, -1);

    std::vector<float> cL, cA, cB, cX, cY;
    cL.reserve(K); cA.reserve(K); cB.reserve(K); 
    cX.reserve(K); cY.reserve(K);

    int k_idx = 0;
    for (float cy = S/2; cy < H; cy += S) {
        for (float cx = S/2; cx < W; cx += S) {
            cL.push_back(L[(int)cy * W + (int)cx]);
            cA.push_back(A[(int)cy * W + (int)cx]);
            cB.push_back(B[(int)cy * W + (int)cx]);
            cX.push_back(cx);
            cY.push_back(cy);

            // 填寫 grid_to_k
            int gx = (int)(cx / S);
            int gy = (int)(cy / S);
            if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
                grid_to_k[gy * grid_w + gx] = k_idx;
            }
            k_idx++;
        }
    }

    int Kactual = cL.size();
    std::vector<int> labels(N, -1);
    std::vector<float> dist(N, FLT_MAX); // 雖然 Pixel-centric 不需要 dist array 來防 race condition，但為了邏輯完整保留

    // 參數設定：與你覺得效果好的 CUDA 版本一致
    const float m = 20.0f; 
    float inv_S2 = 1.0f / (S * S);
    float m2 = m * m;

    // -----------------------------------------------------
    // Step 3: Iterate SLIC loops
    // -----------------------------------------------------
    for (int iter = 0; iter < max_iters; ++iter) {

        // ============================================
        // Assignment Step (Pixel-Centric)
        // ============================================
        // 讓每個像素去搜尋周圍的 Grid Center
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t idx = y * W + x;
                
                float l_i = L[idx];
                float a_i = A[idx];
                float b_i = B[idx];

                float min_D = FLT_MAX;
                int best_k = -1;

                int gx = (int)(x / S);
                int gy = (int)(y / S);

                // 搜尋 3x3 Grid
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ngx = gx + dx;
                        int ngy = gy + dy;

                        if (ngx >= 0 && ngx < grid_w && ngy >= 0 && ngy < grid_h) {
                            int k = grid_to_k[ngy * grid_w + ngx];
                            
                            if (k != -1) {
                                float dL = l_i - cL[k];
                                float dA = a_i - cA[k];
                                float dB = b_i - cB[k];
                                float dcolor = dL*dL + dA*dA + dB*dB;

                                float dX = (float)x - cX[k];
                                float dY = (float)y - cY[k];
                                float dspace = dX*dX + dY*dY;

                                float D = dcolor + (dspace * inv_S2 * m2);

                                if (D < min_D) {
                                    min_D = D;
                                    best_k = k;
                                }
                            }
                        }
                    }
                }
                labels[idx] = best_k;
                // dist[idx] = min_D; // Seq版其實不需要存 dist，只需 label
            }
        }

        // ============================================
        // Update Step
        // ============================================
        std::vector<float> sumL(Kactual, 0), sumA(Kactual, 0), sumB(Kactual, 0);
        std::vector<float> sumX(Kactual, 0), sumY(Kactual, 0);
        std::vector<int> counts(Kactual, 0);

        for (size_t i = 0; i < N; ++i) {
            int k = labels[i];
            if (k != -1) {
                sumL[k] += L[i];
                sumA[k] += A[i];
                sumB[k] += B[i];
                sumX[k] += (float)(i % W); // x
                sumY[k] += (float)(i / W); // y
                counts[k]++;
            }
        }

        for (int k = 0; k < Kactual; ++k) {
            if (counts[k] > 0) {
                float inv = 1.0f / counts[k];
                cL[k] = sumL[k] * inv;
                cA[k] = sumA[k] * inv;
                cB[k] = sumB[k] * inv;
                cX[k] = sumX[k] * inv;
                cY[k] = sumY[k] * inv;
            }
        }

        std::cout << "[SLIC-SEQ] Iter " << iter+1 << "/" << max_iters << " done.\n";
    }

    // -----------------------------------------------------
    // Step 4: Write output
    // -----------------------------------------------------
    std::vector<float> sumR(Kactual, 0), sumG(Kactual, 0), sumB_out(Kactual, 0);
    std::vector<int> countPix(Kactual, 0);

    for (size_t i = 0; i < N; ++i) {
        int k = labels[i];
        if (k != -1) {
            sumR[k] += src->data[i*C + 0];
            sumG[k] += src->data[i*C + 1];
            sumB_out[k] += src->data[i*C + 2];
            countPix[k]++;
        }
    }

    std::vector<unsigned char> avgR(Kactual), avgG(Kactual), avgB(Kactual);
    for (int k = 0; k < Kactual; ++k) {
        if (countPix[k] > 0) {
            avgR[k] = (unsigned char)(sumR[k] / countPix[k]);
            avgG[k] = (unsigned char)(sumG[k] / countPix[k]);
            avgB[k] = (unsigned char)(sumB_out[k] / countPix[k]);
        }
    }

    for (size_t i = 0; i < N; ++i) {
        int k = labels[i];
        if (k != -1) {
            dst->data[i*C + 0] = avgR[k];
            dst->data[i*C + 1] = avgG[k];
            dst->data[i*C + 2] = avgB[k];
        } else {
            dst->data[i*C + 0] = 0; 
            dst->data[i*C + 1] = 0; 
            dst->data[i*C + 2] = 0;
        }
        if (C == 4) dst->data[i*C + 3] = 255;
    }
}


void slic_omp(Image* src, Image* dst, int K, int max_iters)
{
    int max_threads = omp_get_max_threads();
    std::cout << "hello" << std::endl;
    std::cout << "Available OpenMP threads: " << max_threads << std::endl;
    int W = src->width;
    int H = src->height;
    int C = src->channels;
    size_t N = (size_t)W * H;

    if (C < 3) return;

    // -----------------------------------------------------
    // Step 1: Precompute Lab (Parallel)
    // -----------------------------------------------------
    std::vector<float> L(N), A(N), B(N);
    
    #pragma omp parallel for
    for (int i = 0; i < (int)N; ++i) {
        unsigned char r = src->data[i*C + 0];
        unsigned char g = src->data[i*C + 1];
        unsigned char b = src->data[i*C + 2];
        rgb2lab(r, g, b, L[i], A[i], B[i]);
    }

    // -----------------------------------------------------
    // Step 2: Initialize Centers (Sequential is fine here)
    // -----------------------------------------------------
    float S = sqrtf((float)N / K);
    int grid_w = (int)ceil(W / S);
    int grid_h = (int)ceil(H / S);
    std::vector<int> grid_to_k(grid_w * grid_h, -1);

    std::vector<float> cL, cA, cB, cX, cY;
    cL.reserve(K); cA.reserve(K); cB.reserve(K); cX.reserve(K); cY.reserve(K);

    int k_idx = 0;
    for (float cy = S/2; cy < H; cy += S) {
        for (float cx = S/2; cx < W; cx += S) {
            int idx = (int)cy * W + (int)cx;
            cL.push_back(L[idx]);
            cA.push_back(A[idx]);
            cB.push_back(B[idx]);
            cX.push_back(cx);
            cY.push_back(cy);

            int gx = (int)(cx / S);
            int gy = (int)(cy / S);
            if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
                grid_to_k[gy * grid_w + gx] = k_idx;
            }
            k_idx++;
        }
    }

    int Kactual = cL.size();
    std::vector<int> labels(N, -1);
    
    // 參數: 與你覺得好的 CUDA 版本一致
    const float m = 20.0f; 
    float inv_S2 = 1.0f / (S * S);
    float m2 = m * m;

    // -----------------------------------------------------
    // Step 3: Main Loop
    // -----------------------------------------------------
    for (int iter = 0; iter < max_iters; ++iter) {

        // ============================================
        // 3.1 Assignment (Parallel Pixel-Centric)
        // ============================================
        // 這裡可以完全平行化，因為每個 pixel 只寫入自己的 label[i]
        // 不會有 Race Condition，所以不會有油畫感
        #pragma omp parallel for
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int i = y * W + x;
                float l_i = L[i];
                float a_i = A[i];
                float b_i = B[i];

                float min_D = FLT_MAX;
                int best_k = -1;

                int gx = x / (int)S;
                int gy = y / (int)S;

                // 搜尋 3x3 Grid
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ngx = gx + dx;
                        int ngy = gy + dy;

                        if (ngx >= 0 && ngx < grid_w && ngy >= 0 && ngy < grid_h) {
                            int k = grid_to_k[ngy * grid_w + ngx];
                            if (k != -1) {
                                float dL = l_i - cL[k];
                                float dA = a_i - cA[k];
                                float dB = b_i - cB[k];
                                float dcolor = dL*dL + dA*dA + dB*dB;

                                float dX = (float)x - cX[k];
                                float dY = (float)y - cY[k];
                                float dspace = dX*dX + dY*dY;

                                float D = dcolor + (dspace * inv_S2 * m2);

                                if (D < min_D) {
                                    min_D = D;
                                    best_k = k;
                                }
                            }
                        }
                    }
                }
                labels[i] = best_k;
            }
        }

        // ============================================
        // 3.2 Update Centers (Parallel Reduction)
        // ============================================
        // 為了避免 Atomic 的效能損失，我們用 Thread-local storage
        std::vector<float> sumL(Kactual, 0), sumA(Kactual, 0), sumB(Kactual, 0);
        std::vector<float> sumX(Kactual, 0), sumY(Kactual, 0);
        std::vector<int> counts(Kactual, 0);

        #pragma omp parallel
        {
            // 每個執行緒私有的累加器
            std::vector<float> t_L(Kactual, 0), t_A(Kactual, 0), t_B(Kactual, 0);
            std::vector<float> t_X(Kactual, 0), t_Y(Kactual, 0);
            std::vector<int> t_cnt(Kactual, 0);

            #pragma omp for nowait
            for (int i = 0; i < (int)N; ++i) {
                int k = labels[i];
                if (k != -1) {
                    t_L[k] += L[i];
                    t_A[k] += A[i];
                    t_B[k] += B[i];
                    t_X[k] += (float)(i % W);
                    t_Y[k] += (float)(i / W);
                    t_cnt[k]++;
                }
            }

            // 合併回全域變數 (Critical Section)
            #pragma omp critical
            {
                for (int k = 0; k < Kactual; ++k) {
                    sumL[k] += t_L[k];
                    sumA[k] += t_A[k];
                    sumB[k] += t_B[k];
                    sumX[k] += t_X[k];
                    sumY[k] += t_Y[k];
                    counts[k] += t_cnt[k];
                }
            }
        }

        // 計算平均 (這裡運算量極小，不用平行化)
        for (int k = 0; k < Kactual; ++k) {
            if (counts[k] > 0) {
                float inv = 1.0f / counts[k];
                cL[k] = sumL[k] * inv;
                cA[k] = sumA[k] * inv;
                cB[k] = sumB[k] * inv;
                cX[k] = sumX[k] * inv;
                cY[k] = sumY[k] * inv;
            }
        }

        std::cout << "[SLIC-OMP] Iter " << iter+1 << " done.\n";
    }

    // -----------------------------------------------------
    // Step 4: Write Output (Parallel)
    // -----------------------------------------------------
    std::vector<float> sumR(Kactual, 0), sumG(Kactual, 0), sumB_out(Kactual, 0);
    std::vector<int> countPix(Kactual, 0);

    #pragma omp parallel
    {
        std::vector<float> t_R(Kactual, 0), t_G(Kactual, 0), t_B(Kactual, 0);
        std::vector<int> t_cnt(Kactual, 0);

        #pragma omp for nowait
        for (int i = 0; i < (int)N; ++i) {
            int k = labels[i];
            if (k != -1) {
                t_R[k] += src->data[i*C + 0];
                t_G[k] += src->data[i*C + 1];
                t_B[k] += src->data[i*C + 2];
                t_cnt[k]++;
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < Kactual; ++k) {
                sumR[k] += t_R[k];
                sumG[k] += t_G[k];
                sumB_out[k] += t_B[k];
                countPix[k] += t_cnt[k];
            }
        }
    }

    std::vector<unsigned char> avgR(Kactual), avgG(Kactual), avgB(Kactual);
    for (int k = 0; k < Kactual; ++k) {
        if (countPix[k] > 0) {
            avgR[k] = (unsigned char)(sumR[k] / countPix[k]);
            avgG[k] = (unsigned char)(sumG[k] / countPix[k]);
            avgB[k] = (unsigned char)(sumB_out[k] / countPix[k]);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < (int)N; ++i) {
        int k = labels[i];
        if (k != -1) {
            dst->data[i*C + 0] = avgR[k];
            dst->data[i*C + 1] = avgG[k];
            dst->data[i*C + 2] = avgB[k];
        } else {
            dst->data[i*C] = dst->data[i*C+1] = dst->data[i*C+2] = 0;
        }
        if (C == 4) dst->data[i*C + 3] = 255;
    }
}



#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            return; \
        } \
    } while (0)

// ---------------------------------------------------------
// Device Function: RGB to Lab (完全對照你的 Sequential 算法)
// ---------------------------------------------------------
__device__ void rgb2lab_gpu(unsigned char R, unsigned char G, unsigned char B,
                            float &L, float &a, float &b)
{
    float r = R / 255.0f;
    float g = G / 255.0f;
    float bl = B / 255.0f;

    // sRGB to XYZ
    if (r > 0.04045f) r = powf((r + 0.055f) / 1.055f, 2.4f); else r = r / 12.92f;
    if (g > 0.04045f) g = powf((g + 0.055f) / 1.055f, 2.4f); else g = g / 12.92f;
    if (bl > 0.04045f) bl = powf((bl + 0.055f) / 1.055f, 2.4f); else bl = bl / 12.92f;

    float X = r * 0.4124564f + g * 0.3575761f + bl * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + bl * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + bl * 0.9503041f;

    // D65 Normalize
    X /= 0.95047f;
    Z /= 1.08883f;

    auto f2 = [](float t) {
        return (t > 0.008856f) ? powf(t, 1.0f/3.0f) : (7.787f * t + 16.0f/116.0f);
    };
    
    float fx = f2(X);
    float fy = f2(Y);
    float fz = f2(Z);

    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}

// ---------------------------------------------------------
// Kernel: Precompute Lab (每個像素獨立計算)
// ---------------------------------------------------------
__global__ void k_rgb2lab(const unsigned char* __restrict__ src,
                          float* d_L, float* d_A, float* d_B,
                          int W, int H, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    unsigned char r = src[idx * C + 0];
    unsigned char g = src[idx * C + 1];
    unsigned char b = src[idx * C + 2];

    float L, A, B;
    rgb2lab_gpu(r, g, b, L, A, B);

    d_L[idx] = L;
    d_A[idx] = A;
    d_B[idx] = B;
}

// ---------------------------------------------------------
// Kernel: Assignment (關鍵步驟！像素去找中心，保證無競爭)
// ---------------------------------------------------------
__global__ void k_assignment(const float* __restrict__ d_L, 
                             const float* __restrict__ d_A, 
                             const float* __restrict__ d_B,
                             const float* __restrict__ cL,
                             const float* __restrict__ cA,
                             const float* __restrict__ cB,
                             const float* __restrict__ cX,
                             const float* __restrict__ cY,
                             const int* __restrict__ grid_to_k,
                             int* d_labels,
                             int W, int H, int grid_w, int grid_h,
                             float S, float m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int x = idx % W;
    int y = idx / W;

    float l_val = d_L[idx];
    float a_val = d_A[idx];
    float b_val = d_B[idx];

    // 1. 算出這個像素屬於哪個 Grid
    int gx = x / (int)S;
    int gy = y / (int)S;

    float min_dist = FLT_MAX;
    int best_k = -1;
    
    // 預先計算常數，減少除法
    float inv_S2 = 1.0f / (S * S);
    float m2 = m * m;

    // 2. 只搜尋周圍 3x3 的 Grid Center (相當於 Sequntial 的 2S x 2S 範圍)
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int ngx = gx + dx;
            int ngy = gy + dy;

            // 邊界檢查
            if (ngx >= 0 && ngx < grid_w && ngy >= 0 && ngy < grid_h) {
                int k = grid_to_k[ngy * grid_w + ngx];
                
                // 某些邊緣 Grid 可能沒有被分配中心點
                if (k != -1) {
                    float dL = l_val - cL[k];
                    float dA = a_val - cA[k];
                    float dB = b_val - cB[k];
                    float dcolor = dL*dL + dA*dA + dB*dB;

                    float dX = (float)x - cX[k];
                    float dY = (float)y - cY[k];
                    float dspace = dX*dX + dY*dY;

                    // SLIC 標準距離公式
                    float D = dcolor + (dspace * inv_S2 * m2);

                    if (D < min_dist) {
                        min_dist = D;
                        best_k = k;
                    }
                }
            }
        }
    }
    d_labels[idx] = best_k;
}

// ---------------------------------------------------------
// Kernel: Accumulate (使用 Atomic 加法，這是最安全的平行方式)
// ---------------------------------------------------------
__global__ void k_accumulate(const float* __restrict__ d_L,
                             const float* __restrict__ d_A,
                             const float* __restrict__ d_B,
                             const int* __restrict__ d_labels,
                             float* accL, float* accA, float* accB,
                             float* accX, float* accY, int* accCount,
                             int W, int H, int Kactual)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int k = d_labels[idx];
    if (k < 0 || k >= Kactual) return;

    int x = idx % W;
    int y = idx / W;

    atomicAdd(&accL[k], d_L[idx]);
    atomicAdd(&accA[k], d_A[idx]);
    atomicAdd(&accB[k], d_B[idx]);
    atomicAdd(&accX[k], (float)x);
    atomicAdd(&accY[k], (float)y);
    atomicAdd(&accCount[k], 1);
}

// ---------------------------------------------------------
// Kernel: Update Centers (取平均)
// ---------------------------------------------------------
__global__ void k_update_centers(float* cL, float* cA, float* cB,
                                 float* cX, float* cY,
                                 const float* accL, const float* accA, const float* accB,
                                 const float* accX, const float* accY, const int* accCount,
                                 int Kactual)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Kactual) return;

    int count = accCount[k];
    if (count > 0) {
        float inv = 1.0f / count;
        cL[k] = accL[k] * inv;
        cA[k] = accA[k] * inv;
        cB[k] = accB[k] * inv;
        cX[k] = accX[k] * inv;
        cY[k] = accY[k] * inv;
    }
}

// ---------------------------------------------------------
// Kernel: Reconstruction (畫出最終結果)
// ---------------------------------------------------------
__global__ void k_reconstruct(const unsigned char* __restrict__ src,
                              unsigned char* dst,
                              const int* __restrict__ labels,
                              float* sumR, float* sumG, float* sumB, int* count,
                              int W, int H, int C, int Kactual)
{
    // Step 1: Accumulate RGB (Atomic)
    // 為了簡化，這裡將 Accumulate 和 Write 分開寫在同一個 Kernel 會有同步問題
    // 所以我們把 Accumulate RGB 獨立出來，或者重用前面的邏輯。
    // 這裡為了清晰，我們拆成兩個 Kernel 呼叫（見 Host Code），這裡只負責累積。
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    
    int k = labels[idx];
    if(k < 0 || k >= Kactual) return;

    atomicAdd(&sumR[k], (float)src[idx*C+0]);
    atomicAdd(&sumG[k], (float)src[idx*C+1]);
    atomicAdd(&sumB[k], (float)src[idx*C+2]);
    atomicAdd(&count[k], 1);
}

__global__ void k_write_pixels(unsigned char* dst,
                               const int* __restrict__ labels,
                               const float* sumR, const float* sumG, const float* sumB, const int* count,
                               int W, int H, int C, int Kactual)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    
    int k = labels[idx];
    if(k < 0 || k >= Kactual) return;

    int cnt = count[k];
    if(cnt > 0) {
        dst[idx*C+0] = (unsigned char)(sumR[k] / cnt);
        dst[idx*C+1] = (unsigned char)(sumG[k] / cnt);
        dst[idx*C+2] = (unsigned char)(sumB[k] / cnt);
    } else {
        dst[idx*C+0] = 0; dst[idx*C+1] = 0; dst[idx*C+2] = 0;
    }
    if(C==4) dst[idx*C+3] = 255;
}

// ---------------------------------------------------------
// Host Function
// ---------------------------------------------------------
void slic_cuda(Image* src, Image* dst, int K, int max_iters)
{
    int W = src->width;
    int H = src->height;
    int C = src->channels;
    size_t N = (size_t)W * H;

    if (C < 3) { std::cerr << "Requires RGB.\n"; return; }

    // 1. GPU Memory Allocation
    float *d_L, *d_A, *d_B;
    int *d_labels;
    unsigned char *d_src, *d_dst;
    
    CUDA_CHECK(cudaMalloc(&d_L, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src, N * C * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * C * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_src, src->data, N * C * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // 2. Precompute Lab
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    k_rgb2lab<<<numBlocks, blockSize>>>(d_src, d_L, d_A, d_B, W, H, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Initialize Centers (on Host to match logic strictly)
    std::vector<float> h_L(N), h_A(N), h_B(N);
    CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));

    float S = sqrtf((float)N / K);
    
    std::vector<float> h_cL, h_cA, h_cB, h_cX, h_cY;
    h_cL.reserve(K); h_cA.reserve(K); h_cB.reserve(K); h_cX.reserve(K); h_cY.reserve(K);

    int grid_w = (int)ceil(W / S);
    int grid_h = (int)ceil(H / S);
    std::vector<int> h_grid_to_k(grid_w * grid_h, -1);

    int k_idx = 0;
    for (float cy = S/2; cy < H; cy += S) {
        for (float cx = S/2; cx < W; cx += S) {
            size_t ix = (int)cx;
            size_t iy = (int)cy;
            size_t idx = iy * W + ix;
            
            h_cL.push_back(h_L[idx]);
            h_cA.push_back(h_A[idx]);
            h_cB.push_back(h_B[idx]);
            h_cX.push_back(cx);
            h_cY.push_back(cy);

            int gx = (int)(cx / S);
            int gy = (int)(cy / S);
            if (gx < grid_w && gy < grid_h) h_grid_to_k[gy*grid_w + gx] = k_idx;
            
            k_idx++;
        }
    }
    int Kactual = k_idx;

    // Allocate Center Data on GPU
    float *d_cL, *d_cA, *d_cB, *d_cX, *d_cY;
    int *d_grid_to_k;
    float *d_accL, *d_accA, *d_accB, *d_accX, *d_accY;
    int *d_accCount;

    CUDA_CHECK(cudaMalloc(&d_cL, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cA, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cB, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cX, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cY, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_to_k, h_grid_to_k.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_accL, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accA, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accB, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accX, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accY, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accCount, Kactual * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_cL, h_cL.data(), Kactual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), Kactual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), Kactual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cX, h_cX.data(), Kactual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cY, h_cY.data(), Kactual * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_to_k, h_grid_to_k.data(), h_grid_to_k.size() * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Main Loop
    const float m = 20.0f; // ★ 如果你想要更方塊、更不像油畫，可以把這個值調大 (例如 20.0f)
    int blocksK = (Kactual + 255) / 256;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Step A: Assignment (Pixel-Centric) -> 這是產生清晰邊緣的關鍵
        k_assignment<<<numBlocks, blockSize>>>(d_L, d_A, d_B, 
                                               d_cL, d_cA, d_cB, d_cX, d_cY, 
                                               d_grid_to_k, d_labels,
                                               W, H, grid_w, grid_h, S, m);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step B: Reset Accumulators
        CUDA_CHECK(cudaMemset(d_accL, 0, Kactual * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_accA, 0, Kactual * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_accB, 0, Kactual * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_accX, 0, Kactual * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_accY, 0, Kactual * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_accCount, 0, Kactual * sizeof(int)));

        // Step C: Accumulate
        k_accumulate<<<numBlocks, blockSize>>>(d_L, d_A, d_B, d_labels,
                                               d_accL, d_accA, d_accB, d_accX, d_accY, d_accCount,
                                               W, H, Kactual);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step D: Update Centers
        k_update_centers<<<blocksK, 256>>>(d_cL, d_cA, d_cB, d_cX, d_cY,
                                           d_accL, d_accA, d_accB, d_accX, d_accY, d_accCount,
                                           Kactual);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 5. Reconstruction (Reuse accumulators for RGB)
    float *d_sumR = d_accL; 
    float *d_sumG = d_accA; 
    float *d_sumB = d_accB; 
    int *d_cntP = d_accCount;

    CUDA_CHECK(cudaMemset(d_sumR, 0, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sumG, 0, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sumB, 0, Kactual * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cntP, 0, Kactual * sizeof(int)));

    k_reconstruct<<<numBlocks, blockSize>>>(d_src, d_dst, d_labels, d_sumR, d_sumG, d_sumB, d_cntP, W, H, C, Kactual);
    CUDA_CHECK(cudaDeviceSynchronize());

    k_write_pixels<<<numBlocks, blockSize>>>(d_dst, d_labels, d_sumR, d_sumG, d_sumB, d_cntP, W, H, C, Kactual);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy Back
    CUDA_CHECK(cudaMemcpy(dst->data, d_dst, N * C * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_L); cudaFree(d_A); cudaFree(d_B); cudaFree(d_labels);
    cudaFree(d_src); cudaFree(d_dst);
    cudaFree(d_cL); cudaFree(d_cA); cudaFree(d_cB); cudaFree(d_cX); cudaFree(d_cY);
    cudaFree(d_grid_to_k);
    cudaFree(d_accL); cudaFree(d_accA); cudaFree(d_accB); cudaFree(d_accX); cudaFree(d_accY); cudaFree(d_accCount);
}
