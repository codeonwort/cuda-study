#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <time.h>
#include <vector>
#include <array>

#define INPUT_WIDTH    17
#define INPUT_HEIGHT   9
#define INPUT_DEPTH    4
#define STENCIL_ORDER  1
// Order: self, -X, +X, -Y, +Y, -Z, +Z
#define NUM_COEFFS     (1 + 6 * (STENCIL_ORDER))
#define BLOCK_DIM      dim3(16, 16, 1)

// Stencil coefficients
__constant__ int S_device[NUM_COEFFS];

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
	std::vector<ElementType> m; // z-major, then y-major
};

// ------------------------------------------
// Naive stencil

template<typename T>
__global__ void kernel_stencil_naive(
	T* src, int width, int height, int depth,
	T* dst)
{
	int outX = threadIdx.x + (blockDim.x * blockIdx.x);
	int outY = threadIdx.y + (blockDim.y * blockIdx.y);
	int outZ = threadIdx.z + (blockDim.z * blockIdx.z);
	int outIx = (outZ * height * width) + (outY * width) + outX;

	// CUDA even supports lambda? cool
	auto Ix = [&](int x, int y, int z) { return (z * height * width) + (y * width) + x; };

	T ret = T(0);

	bool bInBound = (STENCIL_ORDER <= outX && outX < width - STENCIL_ORDER)
		&& (STENCIL_ORDER <= outY && outY < height - STENCIL_ORDER)
		&& (STENCIL_ORDER <= outZ && outZ < depth - STENCIL_ORDER);

	// x : stencil order
	// Load (1 + 6 * x) input values => (1 + 6 * x) * 4 bytes
	// Perform (1 + 6 * x) MUL and (6 * x) ADD
	// => OP/B = (1 + 12x) / (4 + 24x) // 0.46 for x=1

	if (bInBound) {
		ret = S_device[0] * src[outIx];
		for (int i = 1; i <= STENCIL_ORDER; ++i) {
			ret += S_device[(0 * STENCIL_ORDER) + i] * src[Ix(outX - i, outY, outZ)];
			ret += S_device[(1 * STENCIL_ORDER) + i] * src[Ix(outX + i, outY, outZ)];
			ret += S_device[(2 * STENCIL_ORDER) + i] * src[Ix(outX, outY - i, outZ)];
			ret += S_device[(3 * STENCIL_ORDER) + i] * src[Ix(outX, outY + i, outZ)];
			ret += S_device[(4 * STENCIL_ORDER) + i] * src[Ix(outX, outY, outZ - i)];
			ret += S_device[(5 * STENCIL_ORDER) + i] * src[Ix(outX, outY, outZ + i)];
		}
		dst[outIx] = ret;
	} else if (outX < width && outY < height && outZ < depth) {
		// On boundary
		dst[outIx] = src[outIx];
	}
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
	std::array<ElementType, NUM_COEFFS> S;

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
		p = 0;
		for (int i = 0; i < NUM_COEFFS; ++i) {
			if constexpr (bIntVolume) {
				S[p++] = rand() % 5 - 2;
			}
			if constexpr (bFloatVolume) {
				S[p++] = (float)(rand() % 65536) / 65536.0f;
			}
		}
	}

	// Prepare stencil coefficients
	{
		CUDA_ASSERT(cudaMemcpyToSymbol(S_device, S.data(), S.size() * sizeof(ElementType)));
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
		if (true && bIntVolume) {
			puts("> Input");
			int p = 0;
			for (int z = 0; z < M1.depth; ++z) {
				printf("[slice %d]\n", z);
				for (int y = 0; y < M1.height; ++y) {
					for (int x = 0; x < M1.width; ++x) {
						printf("%d\t", M1.m[p++]);
					}
					puts("");
				}
			}
			printf("> Coeffs: [");
			for (int i = 0; i < NUM_COEFFS; ++i) {
				printf("%d ", S[i]);
			}
			printf("]\n");
			puts("> Output");
			p = 0;
			for (int z = 0; z < M1.depth; ++z) {
				printf("[slice %d]\n", z);
				for (int y = 0; y < M1.height; ++y) {
					for (int x = 0; x < M1.width; ++x) {
						printf("%d\t", M2_naive.m[p++]);
					}
					puts("");
				}
			}
		}
		//for (size_t i = 0; i < M2_naive.m.size(); ++i) {
		//	assert(M1.m[i] == M2_naive.m[i]);
		//}
		//puts("> Result is correct");
	}

	// ------------------------------------------
	// Terminate

	CUDA_ASSERT(cudaDeviceReset());

	puts("cuda destroyed");

	return 0;
}
