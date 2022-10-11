// Chapter 11. Prefix sum (scan)

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <random>

// TODO: Implement Brent-Kung for long input
#define BRENT_KUNG_LONG_INPUT 0

#define CONTENT_TYPE  int32_t
#define CONTENT_COUNT (3840 * 2160)
#define BLOCK_DIM     1024

// TODO: Double buffering
__global__ void kernel_scan_Kogge_Stone(
	CONTENT_TYPE* src, uint32_t totalCount,
	uint32_t* blockCounter, CONTENT_TYPE* scanBlock,
	uint32_t* syncFlag,
	CONTENT_TYPE* dst)
{
	__shared__ uint32_t bid_s;
	__shared__ CONTENT_TYPE XY[BLOCK_DIM];
	__shared__ CONTENT_TYPE XY2[BLOCK_DIM];

	// Dynamic block index assignment
	if (threadIdx.x == 0) {
		bid_s = atomicAdd(blockCounter, 1);
	}
	__syncthreads();
	uint32_t bid = bid_s; // Now use bid instead of blockIdx.x

	// Phase 1. Local scan per block, then store to 'scanBlock'

	uint32_t ix = (bid * BLOCK_DIM) + threadIdx.x;
	uint32_t tid = threadIdx.x;
	if (ix < totalCount) {
		XY[tid] = src[ix];
	} else {
		XY[tid] = CONTENT_TYPE(0);
	}

	bool writeToXY2 = true;
	for (uint32_t stride = 1; stride < BLOCK_DIM; stride *= 2) {
		__syncthreads();

#if 1
		// Double buffering
		if (writeToXY2) {
			if (tid >= stride) {
				XY2[tid] = XY[tid] + XY[tid - stride];
			} else {
				XY2[tid] = XY[tid];
			}
		} else {
			if (tid >= stride) {
				XY[tid] = XY2[tid] + XY2[tid - stride];
			} else {
				XY[tid] = XY2[tid];
			}
		}
		writeToXY2 = !writeToXY2;
#else
		CONTENT_TYPE temp;
		if (tid >= stride) {
			temp = XY[tid] + XY[tid - stride];
		}
		__syncthreads();
		if (tid >= stride) {
			XY[tid] = temp;
		}
#endif
	}
	__syncthreads();
	__shared__ float localSum;
	if (threadIdx.x == blockDim.x - 1) {
		int32_t lastIx = (BLOCK_DIM - 1) - max(0, (int32_t)ix - (int32_t)totalCount);
		if (writeToXY2) {
			localSum = XY[lastIx];
		} else {
			localSum = XY2[lastIx];
		}
		scanBlock[bid] = localSum;
	}

	// Phase 2. Domino-style scan with synchronization across blocks
	__shared__ float prevSum;
	if (threadIdx.x == 0) {
		if (bid != 0) {
			while (atomicAdd(syncFlag, 0) < bid) {}
			prevSum = scanBlock[bid - 1];
			scanBlock[bid] = prevSum + localSum;
			// Memory fence (ensures memory write is visible from other blocks)
			__threadfence();
		}
		atomicAdd(syncFlag, 1);
	}
	__syncthreads();

	// Phase 3. Add prevSum to all elements in current block to produce final result
	if (ix < totalCount) {
		dst[ix] = (writeToXY2 ? XY[tid] : XY2[tid]) + prevSum;
	}
}

#if BRENT_KUNG_LONG_INPUT
#define SECTION_SIZE  2048
__global__ void kernel_scan_Brent_Kung(
	CONTENT_TYPE* src, uint32_t totalCount,
	CONTENT_TYPE* dst)
{
	__shared__ CONTENT_TYPE XY[SECTION_SIZE];

	uint32_t ix = (2 * blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t tid = threadIdx.x;
	if (ix < totalCount) {
		XY[tid] = src[ix];
	}
	if (ix + blockDim.x < totalCount) {
		XY[tid + blockDim.x] = src[ix + blockDim.x];
	}

	// Reduction tree phase
	for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
#if 1
		// Map a continuous section of threads to the XY positions
		// whose indices are of the form k*2^n - 1
		uint32_t p = (tid + 1) * 2 * stride - 1;
		if (p < SECTION_SIZE) {
			XY[p] += XY[p - stride];
		}
#else
		// Causes significant control divergence
		if ((tid + 1) % (2 * stride) == 0) {
			XY[tid] += XY[tid - stride];
		}
#endif
	}
	// Reverse tree phase
	for (uint32_t stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		uint32_t p = (tid + 1) * stride * 2 - 1;
		if (p + stride < SECTION_SIZE) {
			XY[p + stride] += XY[p];
		}
	}
	__syncthreads();
	if (ix < totalCount) {
		dst[ix] = XY[tid];
	}
	if (ix + blockDim.x < totalCount) {
		dst[ix + blockDim.x] = XY[tid + blockDim.x];
	}
}
#endif // BRENT_KUNG_LONG_INPUT

int runTest_scan(int argc, char* argv[])
{
	// ------------------------------------------
	// Query device properties

	int cudaDeviceId;
	CUDA_ASSERT(cudaGetDevice(&cudaDeviceId));

	cudaDeviceProp deviceProps;
	CUDA_ASSERT(cudaGetDeviceProperties(&deviceProps, cudaDeviceId));

	const float KHZ_TO_GHZ = 0.001f * 0.001f;

	puts("CUDA device properties");
	printf("\ttotalConstMem      : %zu bytes\n", deviceProps.totalConstMem);
	// 49152 bytes = 12288 floats or (u)int32
	printf("\tsharedMemPerBlock  : %zu bytes\n", deviceProps.sharedMemPerBlock);
	printf("\twarpSize           : %d\n", deviceProps.warpSize);
	printf("\tclockRate          : %f GHz\n", KHZ_TO_GHZ * (float)deviceProps.clockRate);
	printf("\tmemoryBusWidth     : %d bits\n", deviceProps.memoryBusWidth);
	printf("\tmemoryClockRate    : %f GHz\n", KHZ_TO_GHZ * deviceProps.memoryClockRate);
	printf("\tmaxThreadsPerBlock : %d\n", deviceProps.maxThreadsPerBlock);

	assert(deviceProps.maxThreadsPerBlock >= BLOCK_DIM);

	// ------------------------------------------
	// Host -> device

	std::random_device randDevice;
	std::mt19937 randGen(randDevice());
	//std::uniform_real_distribution<CONTENT_TYPE> randSampler(-1.0f, 1.0f);
	std::uniform_int_distribution<CONTENT_TYPE> randSampler(-128, 128);

	std::vector<CONTENT_TYPE> input(CONTENT_COUNT);
	for (size_t i = 0; i < CONTENT_COUNT; ++i) {
		CONTENT_TYPE x = randSampler(randGen);
		input[i] = x;
	}
	printf("Generate %d input data\n", CONTENT_COUNT);

	std::vector<CONTENT_TYPE> answer(CONTENT_COUNT);
	answer[0] = input[0];
	for (size_t i = 1; i < CONTENT_COUNT; ++i) {
		answer[i] = answer[i - 1] + input[i];
	}

	const size_t contentTotalBytes = sizeof(CONTENT_TYPE) * CONTENT_COUNT;

	// For Kogge-Stone
	const uint32_t scanBlockCount = calcNumBlocks(CONTENT_COUNT, BLOCK_DIM).x;
	const size_t scanBlockTotalBytes = sizeof(CONTENT_TYPE) * scanBlockCount;
	CONTENT_TYPE* content_dev;
	uint32_t* blockCounter_dev;
	CONTENT_TYPE* scanBlock_dev;
	uint32_t* syncFlag_dev;
	CONTENT_TYPE* result_dev;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&blockCounter_dev, sizeof(uint32_t)));
	CUDA_ASSERT(cudaMalloc(&scanBlock_dev, scanBlockTotalBytes));
	CUDA_ASSERT(cudaMalloc(&syncFlag_dev, sizeof(uint32_t)));
	CUDA_ASSERT(cudaMalloc(&result_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMemcpy(content_dev, input.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemset(blockCounter_dev, 0, sizeof(uint32_t)));
	CUDA_ASSERT(cudaMemset(syncFlag_dev, 0, sizeof(uint32_t)));

	// For Brent-Kung
#if BRENT_KUNG_LONG_INPUT
	CONTENT_TYPE* content2_dev;
	CONTENT_TYPE* result2_dev;
	CUDA_ASSERT(cudaMalloc(&content2_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&result2_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMemcpy(content2_dev, input.data(), contentTotalBytes, cudaMemcpyHostToDevice));
#endif

	// ------------------------------------------
	// Kernel: Prefix sum (scan)

	const dim3 blockDim(BLOCK_DIM, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(CONTENT_COUNT, 1, 1), dim3(BLOCK_DIM, 1, 1));

	kernel_scan_Kogge_Stone<<<numBlocks, blockDim>>>(
		content_dev, CONTENT_COUNT,
		blockCounter_dev, scanBlock_dev, syncFlag_dev,
		result_dev);

#if BRENT_KUNG_LONG_INPUT
	const dim3 numBlocks2 = calcNumBlocks(dim3(CONTENT_COUNT / 2, 1, 1), BLOCK_DIM);
	kernel_scan_Brent_Kung<<<numBlocks2, blockDim>>>(
		content2_dev, CONTENT_COUNT,
		result2_dev);
#endif

	// ------------------------------------------
	// Device -> host

	// Result of Kogge-Stone
	std::vector<CONTENT_TYPE> result(CONTENT_COUNT);
	//std::vector<CONTENT_TYPE> scanBlock(numBlocks.x);
	CUDA_ASSERT(cudaMemcpy(result.data(), result_dev, contentTotalBytes, cudaMemcpyDeviceToHost));
	//CUDA_ASSERT(cudaMemcpy(scanBlock.data(), scanBlock_dev, scanBlockTotalBytes, cudaMemcpyDeviceToHost));

#if BRENT_KUNG_LONG_INPUT
	std::vector<CONTENT_TYPE> result2(CONTENT_COUNT);
	CUDA_ASSERT(cudaMemcpy(result2.data(), result2_dev, contentTotalBytes, cudaMemcpyDeviceToHost));
#endif

	puts("Compare results...");
	{
		constexpr uint32_t NUM_SHOW = 8;
		printf("input: [");
		for (size_t i = 0; i < result.size(); ++i) {
			if (i < NUM_SHOW) {
				printf("%d ", input[i]);
			} else if (i == NUM_SHOW) {
				printf("...]\n");
			}
		}
		printf("scan : [");
		for (size_t i = 0; i < result.size(); ++i) {
			CONTENT_TYPE dx = std::abs(answer[i] - result[i]);
			assert(dx == 0);

#if BRENT_KUNG_LONG_INPUT
			CONTENT_TYPE dx2 = std::abs(answer[i] - result2[i]);
			assert(dx2 == 0);
#endif

			if (i < NUM_SHOW) {
				printf("%d ", answer[i]);
			} else if (i == NUM_SHOW) {
				printf("...");
			}
		}
		puts("]");
	}

	return 0;
}
