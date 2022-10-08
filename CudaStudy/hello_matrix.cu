// Demo 2: Matrix multiplication

#include "tests.cuh"

#include <cuda_runtime.h>
// For threadIdx, blockIdx, blockDim
#include <device_launch_parameters.h>

#include <time.h>
#include <stdlib.h>
#include <vector>

struct Matrix {
    Matrix(int inRows, int inCols)
        : rows(inRows)
        , cols(inCols)
    {
        m.resize(rows * cols);
    }

    size_t sizeInBytes() {
        return rows * cols * sizeof(float);
    }

    int rows;
    int cols;
    std::vector<float> m;
};

// KERNEL: naive matrix mult
// 
// - 2 operations (add and mul)
// - 2 memory access (4 bytes each -> total 8 bytes)
//   (2 FLOP / 8 B = 0.25 FLOP / B)
// 
// - Ampere A100 GPU has a peak global memory bandwidth of 1555 GB/second
// - Matrix mul is limited to 389 GFLOPS (= 1555 GB/s * 0.25 FLOB/B)
// - A100's peak single-precision operation throughput is 19500 GFLOPS
//   (389 GFLOPS is only 2% of 19500 GFLOPS)
//
// - M1, M2: input (row1 x col1, row2 x col2) where row2 = col1
// - M3: output (row1 x col2)
//
__global__ void kernel_matMul_naive(
    float* M1, float* M2, float* M3,
    int rows1, int cols1, int cols2)
{
    const int rows3 = rows1;
    const int cols3 = cols2;

    int elemX = threadIdx.x + (blockDim.x * blockIdx.x);
    int elemY = threadIdx.y + (blockDim.y * blockIdx.y);
    int elemArrayIx = elemX + (elemY * cols3);

    if (elemX >= cols3 || elemY >= rows3) {
        return;
    }

    float ret = 0.0f;
    for (int k = 0; k < cols1; ++k) {
        int ix1 = k + elemY * cols1; // M1: row=elemY, col=k
        int ix2 = elemX + k * cols2; // M2: row=k, col=elemX
        ret += M1[ix1] * M2[ix2];
    }

    M3[elemArrayIx] = ret;
}

// KERNEL: tiled matrix mult
//
// Uses shared memory to reduce traffic to the global memory.
#if 0 // A version that only works for square matrices.
#define TILE_WIDTH 16
__global__ void kernel_matMul_tiled(
    float* M1, float* M2, float* M3,
    int width)
{
    __shared__ float M1_lds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float M2_lds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float ret = 0.0f;
    int maxPh = ceil(width / (float)TILE_WIDTH);
    for (int ph = 0; ph < maxPh; ++ph) {
        // M1 (x,y) = (ph * TILE_WIDTH + tx, row);
        // M2 (x,y) = (col, ph * TILE_WIDTH + ty);
        if ((row < width) && (ph * TILE_WIDTH + tx) < width) {
            M1_lds[ty][tx] = M1[row * width + ph * TILE_WIDTH + tx];
        } else {
            M1_lds[ty][tx] = 0.0f;
        }
        if ((ph * TILE_WIDTH + ty) < width && col < width) {
            M2_lds[ty][tx] = M2[(ph * TILE_WIDTH + ty) * width + col];
        } else {
            M2_lds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            ret += M1_lds[ty][k] * M2_lds[k][tx];
        }
        __syncthreads();
    }

    if (row < width && col < width) {
        M3[row * width + col] = ret;
    }
}
#else
#define TILE_WIDTH 16
__global__ void kernel_matMul_tiled(
    float* M1, float* M2, float* M3,
    int rows1, int cols1, int cols2)
{
    const int rows2 = cols1;
    const int rows3 = rows1;
    const int cols3 = cols2;

    __shared__ float M1_lds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float M2_lds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float ret = 0.0f;
    int maxPh = ceil(cols1 / (float)TILE_WIDTH);
    for (int ph = 0; ph < maxPh; ++ph) {
        int M1_x = ph * TILE_WIDTH + tx;
        int M1_y = row;
        if (M1_y < rows1 && M1_x < cols1) {
            M1_lds[ty][tx] = M1[M1_y * cols1 + M1_x];
        } else {
            M1_lds[ty][tx] = 0.0f;
        }

        int M2_x = col;
        int M2_y = ph * TILE_WIDTH + ty;
        if (M2_y < rows2 && M2_x < cols2) {
            M2_lds[ty][tx] = M2[M2_y * cols2 + M2_x];
        } else {
            M2_lds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            ret += M1_lds[ty][k] * M2_lds[k][tx];
        }
        __syncthreads();
    }

    if (row < rows3 && col < cols3) {
        M3[row * cols3 + col] = ret;
    }
}
#endif


int runTest_matMul(int argc, char* argv[]) {
    // ------------------------------------------
    // Query device properties

    int cudaDeviceId;
    CUDA_ASSERT(cudaGetDevice(&cudaDeviceId));

    cudaDeviceProp devProp;
    CUDA_ASSERT(cudaGetDeviceProperties(&devProp, cudaDeviceId));

    puts("CUDA device properties");
    // 49152 bytes = 48 KiB
    printf("\tsharedMemPerBlock: %zu bytes\n", devProp.sharedMemPerBlock);

    // ------------------------------------------
    // Kernel: matrix mul

    //int rows1 = 5000, cols1 = 4000, cols2 = 6000;
    int rows1 = 100, cols1 = 72, cols2 = 60;
    //int rows1 = 200, cols1 = 200, cols2 = 200;
    if (argc > 4) {
        sscanf_s(argv[1], "%d", &rows1);
        sscanf_s(argv[2], "%d", &cols1);
        sscanf_s(argv[3], "%d", &cols2);
    }

    // Prepare random matrices
    Matrix M1(rows1, cols1);
    Matrix M2(cols1, cols2);
    Matrix M3_naive(M1.rows, M2.cols);
    Matrix M3_tiled(M1.rows, M2.cols);
    {
        srand((unsigned int)time(NULL));
        int p = 0;
        for (int y = 0; y < M1.rows; ++y) {
            for (int x = 0; x < M1.cols; ++x) {
                M1.m[p++] = (float)(rand() % 65536) / 65536.0f;
            }
        }
        p = 0;
        for (int y = 0; y < M2.rows; ++y) {
            for (int x = 0; x < M2.cols; ++x) {
                M2.m[p++] = (float)(rand() % 65536) / 65536.0f;
            }
        }
    }
    assert(M1.cols == M2.rows);

    printf("Matrices (row x col)\n");
    printf("\tM1: %d x %d\n", M1.rows, M1.cols);
    printf("\tM2: %d x %d\n", M2.rows, M2.cols);
    printf("\tM3: %d x %d\n", M3_naive.rows, M3_naive.cols);

    // Naive
    puts("Run kernel: naive matrix mult");
    {
        Matrix& M3 = M3_naive;

        float* M1_device;
        float* M2_device;
        CUDA_ASSERT(cudaMalloc(&M1_device, M1.sizeInBytes()));
        CUDA_ASSERT(cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMalloc(&M2_device, M2.sizeInBytes()));
        CUDA_ASSERT(cudaMemcpy(M2_device, M2.m.data(), M2.sizeInBytes(), cudaMemcpyHostToDevice));

        float* M3_device;
        CUDA_ASSERT(cudaMalloc(&M3_device, M3.sizeInBytes()));

        const dim3 blockSize(16, 16, 1);
        const dim3 numBlocks = calcNumBlocks(dim3(M3.cols, M3.rows, 1), blockSize);
        kernel_matMul_naive<<<numBlocks, blockSize>>>(
            M1_device, M2_device, M3_device,
            M1.rows, M1.cols, M2.cols);

        CUDA_ASSERT(cudaMemcpy(M3.m.data(), M3_device, M3.sizeInBytes(), cudaMemcpyDeviceToHost));

        CUDA_ASSERT(cudaFree(M1_device));
        CUDA_ASSERT(cudaFree(M2_device));
        CUDA_ASSERT(cudaFree(M3_device));
    }

    // Tiled
    puts("Run kernel: tiled matrix mult");
    {
        Matrix& M3 = M3_tiled;

        float* M1_device;
        float* M2_device;
        CUDA_ASSERT(cudaMalloc(&M1_device, M1.sizeInBytes()));
        CUDA_ASSERT(cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMalloc(&M2_device, M2.sizeInBytes()));
        CUDA_ASSERT(cudaMemcpy(M2_device, M2.m.data(), M2.sizeInBytes(), cudaMemcpyHostToDevice));

        float* M3_device;
        CUDA_ASSERT(cudaMalloc(&M3_device, M3.sizeInBytes()));

        const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
        const int numBlocksX = (M3.cols / blockSize.x) + (M3.cols % blockSize.x == 0 ? 0 : 1);
        const int numBlocksY = (M3.rows / blockSize.y) + (M3.rows % blockSize.y == 0 ? 0 : 1);
        const dim3 numBlocks(numBlocksX, numBlocksY, 1);
        kernel_matMul_tiled<<<numBlocks, blockSize>>>(
            M1_device, M2_device, M3_device,
            M1.rows, M1.cols, M2.cols);

        CUDA_ASSERT(cudaMemcpy(M3.m.data(), M3_device, M3.sizeInBytes(), cudaMemcpyDeviceToHost));

        CUDA_ASSERT(cudaFree(M1_device));
        CUDA_ASSERT(cudaFree(M2_device));
        CUDA_ASSERT(cudaFree(M3_device));
    }

    // ------------------------------------------
    // Compare results

    puts("Compare results...");

    for (size_t i = 0; i < M3_naive.m.size(); ++i) {
        if (M3_naive.m[i] != M3_tiled.m[i]) {
            __debugbreak();
        }
    }

    puts("> Results are same");

    // ------------------------------------------

    CUDA_ASSERT(cudaDeviceReset());

    puts("cuda destroyed");

    return 0;
}
