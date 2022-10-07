#include "tests.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

// Special case: Fixed data for 4x4 (see below)
#define INPUT_DATA_WIDTH 326
#define INPUT_DATA_HEIGHT 612

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

// ------------------------------------------
// Naive convolution filter

template<typename T>
__global__ void kernel_convFilter_naive(
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
				//int fIx = fy * (filterRadius * 2 + 1) + fx;

				//ret += src[neighborIx] * F[fIx];
				ret += src[neighborIx] * F_device[fy][fx];
			}
		}
	}

	if (outX < width && outY < height) {
		dst[outIx] = ret;
	}
}

// ------------------------------------------
// Tiled convolution filter

#define IN_TILE_DIM 16
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (filterRadius))

template<typename T>
__global__ void kernel_convFilter_tiled(
	T* src, int width, int height, T* dst)
{
	constexpr bool bIntMatrix = std::is_same<T, int>::value;
	constexpr bool bFloatMatrix = std::is_same<T, float>::value;
	static_assert(bIntMatrix || bFloatMatrix, "ElementType should be int or float");

	int outX = blockIdx.x * OUT_TILE_DIM + threadIdx.x - filterRadius;
	int outY = blockIdx.y * OUT_TILE_DIM + threadIdx.y - filterRadius;

	bool bValidInput = (0 <= outY && outY < height && 0 <= outX && outX < width);

	// Load input tile
	__shared__ T N_s[IN_TILE_DIM][IN_TILE_DIM];
	if (bValidInput) {
		N_s[threadIdx.y][threadIdx.x] = src[outY * width + outX];
	} else {
		N_s[threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();

	// Calc output elements
	int tileCol = threadIdx.x - filterRadius;
	int tileRow = threadIdx.y - filterRadius;

	bool bValidTile = (0 <= tileCol && tileCol < OUT_TILE_DIM
		&& 0 <= tileRow && tileRow < OUT_TILE_DIM);

	if (bValidInput && bValidTile) {
		T ret = 0;
		for (int fRow = 0; fRow < 2 * filterRadius + 1; ++fRow) {
			for (int fCol = 0; fCol < 2 * filterRadius + 1; ++fCol) {
				ret += F_device[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
			}
		}
		dst[outY * width + outX] = ret;
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

	Matrix<ElementType> M1(INPUT_DATA_HEIGHT, INPUT_DATA_WIDTH); // input
	Matrix<ElementType> F(radius * 2 + 1, radius * 2 + 1); // filter
	Matrix<ElementType> M2_naive(M1.rows, M1.cols); // output
	Matrix<ElementType> M2_tiled(M1.rows, M1.cols); // output

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

#if INPUT_DATA_WIDTH == 4 && INPUT_DATA_HEIGHT == 4
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
	// M2.m should be [-14 -34 -22 -16 -7 -19 -24 -10 -5 -12 -19 -5 0 -12 -12 4]
#endif

	printf("Matrices (row x col)\n");
	printf("\tM1: %d x %d\n", M1.rows, M1.cols);
	printf("\tF: %d x %d\n", F.rows, F.cols);

	// Prepare filter
	{
		//ElementType* F_device;
		//cudaMalloc(&F_device, F.sizeInBytes());
		//cudaMemcpy(F_device, F.m.data(), F.sizeInBytes(), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpyToSymbol(F_device, F.m.data(), F.sizeInBytes());
		assert(cudaStatus == cudaError::cudaSuccess);
	}

	puts("Run kernel: naive convolution filter");
	{
		Matrix<ElementType>& M2 = M2_naive;

		ElementType* M1_device;
		ElementType* M2_device;
		cudaMalloc(&M1_device, M1.sizeInBytes());
		cudaMalloc(&M2_device, M1.sizeInBytes());
		cudaStatus = cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);

		const dim3 blockSize(16, 16, 1);
		const dim3 numBlocks = calcNumBlocks(M1.getDim3(), blockSize);
		kernel_convFilter_naive<<<numBlocks, blockSize>>>(
			M1_device, M1.cols, M1.rows,
			//F_device, radius,
			M2_device);

		cudaStatus = cudaMemcpy(M2.m.data(), M2_device, M2.sizeInBytes(), cudaMemcpyDeviceToHost);
		assert(cudaStatus == cudaError::cudaSuccess);

		cudaFree(M1_device);
		cudaFree(M2_device);
	}

	puts("Run kernel: tiled convolution filter");
	{
		Matrix<ElementType>& M2 = M2_tiled;

		ElementType* M1_device;
		ElementType* M2_device;
		cudaMalloc(&M1_device, M1.sizeInBytes());
		cudaMalloc(&M2_device, M1.sizeInBytes());
		cudaStatus = cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice);
		assert(cudaStatus == cudaError::cudaSuccess);

		// CAUTION: Compute num blocks with outBlockSize,
		//          but actual block size is inBlockSize.
		const dim3 inBlockSize(IN_TILE_DIM, IN_TILE_DIM, 1);
		const dim3 outBlockSize(OUT_TILE_DIM, OUT_TILE_DIM, 1);
		dim3 inputDim = M1.getDim3();
		inputDim.x += filterRadius * 2;
		inputDim.y += filterRadius * 2;
		const dim3 numBlocks = calcNumBlocks(inputDim, outBlockSize);
		kernel_convFilter_tiled<<<numBlocks, inBlockSize>>>(
			M1_device, M1.cols, M1.rows,
			M2_device);

		cudaStatus = cudaMemcpy(M2.m.data(), M2_device, M2.sizeInBytes(), cudaMemcpyDeviceToHost);
		assert(cudaStatus == cudaError::cudaSuccess);

		cudaFree(M1_device);
		cudaFree(M2_device);
	}

	cudaFree(F_device);

	puts("Compare results...");
	{
		for (size_t i = 0; i < M2_naive.m.size(); ++i) {
			if (M2_naive.m[i] != M2_tiled.m[i]) {
				if constexpr (bIntMatrix) {
					fprintf(stderr, "i=%zu naive=%d tiled=%d\n", i, M2_naive.m[i], M2_tiled.m[i]);
				}
				__debugbreak();
			}
		}
		puts("> Results are same");
	}

	// ------------------------------------------

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	puts("cuda destroyed");

	return 0;
}
