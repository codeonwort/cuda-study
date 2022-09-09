#include "tests.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

dim3 calcNumBlocks(const dim3& dimensions, const dim3& blockSize) {
	auto iceil = [](int x, int y) { return (x % y) ? x / y + 1 : x / y; };
	int x = iceil(dimensions.x, blockSize.x);
	int y = iceil(dimensions.y, blockSize.y);
	int z = iceil(dimensions.z, blockSize.z);
	return dim3(x, y, z);
}

template<typename ElementType>
struct Matrix {
	Matrix(int inRows, int inCols)
		: rows(inRows)
		, cols(inCols)
	{
		m.resize(rows * cols);
	}

	size_t sizeInBytes() {
		return rows * cols * sizeof(ElementType);
	}

	dim3 getDim3() { return dim3(cols, rows, 1); }

	int rows;
	int cols;
	std::vector<ElementType> m;
};


#define filterRadius 1
__constant__ int F_device[2 * filterRadius + 1][2 * filterRadius + 1];

// KERNEL: convlution filter
template<typename T>
__global__ void kernel_convFilter(
	T* src, int width, int height,
	//T* filter, int filterRadius, // Now accessed by constant memory
	T* dst)
{
	constexpr bool bIntMatrix = std::is_same<T, int>::value;
	constexpr bool bFloatMatrix = std::is_same<T, float>::value;
	static_assert(bIntMatrix || bFloatMatrix, "ElementType should be int or float");

	int outX = threadIdx.x + (blockDim.x * blockIdx.x);
	int outY = threadIdx.y + (blockDim.y * blockIdx.y);
	int outIx = outY * width + outX;

	T ret = 0;
	for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
		for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX) {
			int neighborX = outX + offsetX;
			int neighborY = outY + offsetY;
			if (0 <= neighborX && neighborX < width && 0 <= neighborY && neighborY < height) {
				int neighborIx = neighborY * width + neighborX;

				int fx = filterRadius + offsetX;
				int fy = filterRadius + offsetY;
				int fIx = fy * (filterRadius * 2 + 1) + fx;

				//ret += src[neighborIx] * F[fIx];
				ret += src[neighborIx] * F_device[fy][fx];
			}
		}
	}

	if (outX < width && outY < height) {
		dst[outIx] = ret;
	}
}

int runTest_convFilter(int argc, char* argv[])
{
	cudaError_t cudaStatus;

	puts("hello, convolution filter");

	// ------------------------------------------
	// Query device properties

	int cudaDeviceId;
	cudaStatus = cudaGetDevice(&cudaDeviceId);
	assert(cudaStatus == cudaError::cudaSuccess);

	cudaDeviceProp devProp;
	cudaStatus = cudaGetDeviceProperties(&devProp, cudaDeviceId);
	assert(cudaStatus == cudaError::cudaSuccess);

	puts("CUDA device properties");
	// 49152 bytes = 48 KiB
	printf("\ttotalConstMem: %zu bytes\n", devProp.totalConstMem);

	// ------------------------------------------
	// Kernel: convolution filter

	int radius = 1;

	using ElementType = int;
	constexpr bool bIntMatrix = std::is_same<ElementType, int>::value;
	constexpr bool bFloatMatrix = std::is_same<ElementType, float>::value;
	static_assert(bIntMatrix || bFloatMatrix, "ElementType should be int or float");

	Matrix<ElementType> M1(4, 4); // input
	Matrix<ElementType> F(radius * 2 + 1, radius * 2 + 1); // filter
	Matrix<ElementType> M2(M1.rows, M1.cols); // output

	// Prepare random matrices
	{
		srand((unsigned int)time(NULL));
		int p = 0;
		for (int y = 0; y < M1.rows; ++y) {
			for (int x = 0; x < M1.cols; ++x) {
				if constexpr (bIntMatrix) {
					M1.m[p++] = rand() % 128;
				}
				if constexpr (bFloatMatrix) {
					M1.m[p++] = (float)(rand() % 65536) / 65536.0f;
				}
			}
		}
		p = 0;
		for (int y = 0; y < radius * 2 + 1; ++y) {
			for (int x = 0; x < radius * 2 + 1; ++x) {
				if constexpr (bIntMatrix) {
					F.m[p++] = rand() % 7 - 3;
				}
				if constexpr (bFloatMatrix) {
					F.m[p++] = (float)(rand() % 65536) / 65536.0f;
				}
			}
		}
	}

#if 1
	int m1[] = {
		5, 1, 6, 4,
		4, 4, 4, 4,
		4, 0, 2, 6,
		5, 2, 1, 5
	};
	int f[] = {
		0, 1, 0,
		-2, 0, -2,
		0, -1, -2
	};
	memcpy(M1.m.data(), m1, sizeof(int) * 16);
	memcpy(F.m.data(), f, sizeof(int) * 9);
#endif

	printf("Matrices (row x col)\n");
	printf("\tM1: %d x %d\n", M1.rows, M1.cols);
	printf("\tF: %d x %d\n", F.rows, F.cols);

	puts("Run kernel: convolution filter");
	{
		ElementType* M1_device;
		//ElementType* F_device;
		ElementType* M2_device;
		cudaMalloc(&M1_device, M1.sizeInBytes());
		//cudaMalloc(&F_device, F.sizeInBytes());
		cudaMalloc(&M2_device, M1.sizeInBytes());
		cudaStatus = cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);
		//cudaMemcpy(F_device, F.m.data(), F.sizeInBytes(), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpyToSymbol(F_device, F.m.data(), F.sizeInBytes());
		assert(cudaStatus == cudaError::cudaSuccess);
		cudaStatus = cudaMemcpy(M2_device, M2.m.data(), M2.sizeInBytes(), cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);

		const dim3 blockSize(16, 16, 1);
		const dim3 numBlocks = calcNumBlocks(M1.getDim3(), blockSize);
		kernel_convFilter<<<numBlocks, blockSize>>>(
			M1_device, M1.cols, M1.rows,
			//F_device, radius,
			M2_device);

		cudaStatus = cudaMemcpy(M2.m.data(), M2_device, M2.sizeInBytes(), cudaMemcpyDeviceToHost);
		assert(cudaStatus == cudaError::cudaSuccess);

		cudaFree(M1_device);
		//cudaFree(F_device);
		cudaFree(M2_device);
	}

	puts("Verify");
	{
		//
	}

	return 0;
}
