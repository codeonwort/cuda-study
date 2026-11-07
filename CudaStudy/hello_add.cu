// Demo 1: Vector add

#include "tests.cuh"

#include <cuda_runtime.h>
// For threadIdx, blockIdx, blockDim
#include <device_launch_parameters.h>

#include <time.h>
#include <stdlib.h>
#include <vector>

#define N 100

/**
 * - In CUDA C, a kernel function specifies the code to be executed
 *   by all threads during a parallel phase.
 * - When a kernel is called, CUDA launches a grid of threads that are
 *   organized into a two-level hierarchy.
 * - Each grid is organized as an array of thread blocks.
 * - Each block can contain up to 1024 threads on current systems.
 * - 'blockDim'  = gl_WorkGroupSize = (local_size_x, local_size_y, local_size_z)
 * - 'threadIdx' = gl_LocalInvocationID
 * - 'blockIdx'  = gl_WorkGroupID
 */
__global__ void kernel_vecAdd(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int runTest_vecAdd(int argc, char* argv[])
{
    cudaError_t cudaStatus;

    puts("hello, cuda");

    float xs[N], ys[N], zs[N];
    for (int i = 0; i < N; ++i) {
        xs[i] = float(i) + 1.0f;
        ys[i] = float(i);
        zs[i] = 0.0f;
    }

    // ------------------------------------------

    const size_t totalBytes = sizeof(float) * N;
    float *g_xs, *g_ys, *g_zs;
    cudaStatus = cudaMalloc(&g_xs, totalBytes);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&g_ys, totalBytes);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&g_zs, totalBytes);
    assert(cudaStatus == cudaSuccess);

    puts("cudaMalloc");

    // ------------------------------------------

    cudaStatus = cudaMemcpy(g_xs, xs, totalBytes, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpy(g_ys, ys, totalBytes, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);

    puts("cudaMemcpy");

    // ------------------------------------------
    // Kernel: vector add

    {
        int numBlocks = (int)(N / 256) + (N % 256 == 0 ? 0 : 1);
        int blockSize = 256;
        kernel_vecAdd<<<numBlocks, blockSize>>> (g_xs, g_ys, g_zs, N);

        cudaMemcpy(zs, g_zs, totalBytes, cudaMemcpyDeviceToHost);
    }

    puts("run Add kernel");

    // ------------------------------------------

    cudaStatus = cudaFree(g_xs);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(g_ys);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(g_zs);
    assert(cudaStatus == cudaSuccess);

    puts("cudaFree");

    // ------------------------------------------

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    puts("cuda destroyed");

	return 0;
}
