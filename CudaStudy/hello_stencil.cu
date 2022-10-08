#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <time.h>
#include <vector>

#define INPUT_WIDTH    19
#define INPUT_HEIGHT   27
#define INPUT_DEPTH    15
#define BLOCK_DIM      dim3(16, 16, 1)

static dim3 calcNumBlocks(const dim3& dimensions, const dim3& blockSize) {
	auto iceil = [](int x, int y) { return (x % y) ? x / y + 1 : x / y; };
	int x = iceil(dimensions.x, blockSize.x);
	int y = iceil(dimensions.y, blockSize.y);
	int z = iceil(dimensions.z, blockSize.z);
	return dim3(x, y, z);
}

template<typename ElementType>
struct Volume {
	Volume(int inWidth, int inHeight, int inDepth)
		: width(inWidth)
		, height(inHeight)
		, depth(inDepth)
	{
		m.resize(width * height * depth);
	}

	size_t sizeInBytes() {
		return width * height * depth * sizeof(ElementType);
	}

	dim3 getDim3() { return dim3(width, height, depth); }

	int width;
	int height;
	int depth;
	std::vector<ElementType> m;
};

// ------------------------------------------
// Naive stencil

template<typename T>
__global__ void kernel_stencil_naive(
	T* src, int rows, int cols, int depth,
	T* dst)
{
	int outX = threadIdx.x + (blockDim.x * blockIdx.x);
	int outY = threadIdx.y + (blockDim.y * blockIdx.y);
	int outZ = threadIdx.z + (blockDim.z * blockIdx.z);
	int outIx = (outZ * cols * rows) + (outY * cols) + outX;
	if (outX > cols || outY > rows || outZ > depth) {
		return;
	}

	T ret = T(0);

	// TODO: Stencil here
	ret = src[outIx];

	dst[outIx] = ret;
}

int runTest_stencil(int argc, char* argv[])
{
	// ------------------------------------------
	// Query device properties

	int cudaDeviceId;
	CUDA_ASSERT(cudaGetDevice(&cudaDeviceId));

	cudaDeviceProp deviceProps;
	CUDA_ASSERT(cudaGetDeviceProperties(&deviceProps, cudaDeviceId));

	puts("CUDA device properties");
	// 49152 bytes = 48 KiB
	printf("\ttotalConstMem: %zu bytes\n", deviceProps.totalConstMem);

	// ------------------------------------------
	// Kernel: stencil

	using ElementType = int;
	constexpr bool bIntVolume = std::is_same<ElementType, int>::value;
	constexpr bool bFloatVolume = std::is_same<ElementType, float>::value;
	static_assert(bIntVolume || bFloatVolume, "ElementType should be int or float");

	Volume<ElementType> M1(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH);
	Volume<ElementType> M2_naive(M1.width, M1.height, M1.depth);
	Volume<ElementType> M2_tiled(M1.width, M1.height, M1.depth);

	// Generate random input
	printf("Input: %dx%dx%d volume\n", M1.width, M1.height, M1.depth);
	{
		srand((unsigned int)time(NULL));
		int p = 0;
		for (int z = 0; z < M1.depth; ++z) {
			for (int y = 0; y < M1.height; ++y) {
				for (int x = 0; x < M1.width; ++x) {
					if constexpr (bIntVolume) {
						M1.m[p++] = rand() % 128;
					}
					if constexpr (bFloatVolume) {
						M1.m[p++] = (float)(rand() % 65536) / 65536.0f;
					}
				}
			}
		}
	}

	puts("Run kernel: naive stencil");
	{
		Volume<ElementType>& M2 = M2_naive;

		ElementType* M1_device;
		ElementType* M2_device;
		CUDA_ASSERT(cudaMalloc(&M1_device, M1.sizeInBytes()));
		CUDA_ASSERT(cudaMalloc(&M2_device, M2.sizeInBytes()));
		CUDA_ASSERT(cudaMemcpy(M1_device, M1.m.data(), M1.sizeInBytes(), cudaMemcpyHostToDevice));

		const dim3 numBlocks = calcNumBlocks(M1.getDim3(), BLOCK_DIM);
		kernel_stencil_naive<<<numBlocks, BLOCK_DIM >>>(
			M1_device, M1.width, M1.height, M1.depth,
			M2_device);

		CUDA_ASSERT(cudaMemcpy(M2.m.data(), M2_device, M2.sizeInBytes(), cudaMemcpyDeviceToHost));
		CUDA_ASSERT(cudaFree(M1_device));
		CUDA_ASSERT(cudaFree(M2_device));
	}

	puts("Verify result");
	{
		for (size_t i = 0; i < M2_naive.m.size(); ++i) {
			assert(M1.m[i] == M2_naive.m[i]);
		}
		puts("> Result is correct");
	}

	// ------------------------------------------
	// Terminate

	CUDA_ASSERT(cudaDeviceReset());

	puts("cuda destroyed");

	return 0;
}
