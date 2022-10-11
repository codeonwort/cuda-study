// Chapter 11. Prefix sum (scan)

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <random>

#define CONTENT_TYPE  int32_t
// TODO: Only works if CONTENT_COUNT <= BLOCK_DIM
#define CONTENT_COUNT 1000
#define BLOCK_DIM     1024

// TODO: Double buffering
__global__ void kernel_scan_Kogge_Stone(
	CONTENT_TYPE* src, uint32_t totalCount,
	CONTENT_TYPE* dst)
{
	__shared__ CONTENT_TYPE XY[BLOCK_DIM];

	uint32_t ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t tid = threadIdx.x;
	if (ix < totalCount) {
		XY[tid] = src[ix];
	} else {
		XY[tid] = CONTENT_TYPE(0);
	}

	for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		CONTENT_TYPE temp;
		if (tid >= stride) {
			temp = XY[tid] + XY[tid - stride];
		}
		__syncthreads();
		if (tid >= stride) {
			XY[tid] = temp;
		}
	}
	if (ix < totalCount) {
		dst[ix] = XY[tid];
	}
}

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
	// CUDA gives you all these info!?
	printf("\ttotalConstMem      : %zu bytes\n", deviceProps.totalConstMem);
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

	CONTENT_TYPE* content_dev;
	CONTENT_TYPE* content2_dev;
	CONTENT_TYPE* result_dev;
	CONTENT_TYPE* result2_dev;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&content2_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&result_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&result2_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMemcpy(content_dev, input.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(content2_dev, input.data(), contentTotalBytes, cudaMemcpyHostToDevice));

	// ------------------------------------------
	// Kernel: Prefix sum (scan)

	const dim3 blockDim(BLOCK_DIM, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(CONTENT_COUNT, 1, 1), BLOCK_DIM);
	kernel_scan_Kogge_Stone<<<numBlocks, blockDim>>>(
		content_dev, CONTENT_COUNT,
		result_dev);

	const dim3 numBlocks2 = calcNumBlocks(dim3(CONTENT_COUNT / 2, 1, 1), BLOCK_DIM);
	kernel_scan_Brent_Kung<<<numBlocks2, blockDim>>>(
		content2_dev, CONTENT_COUNT,
		result2_dev);

	// ------------------------------------------
	// Device -> host

	std::vector<CONTENT_TYPE> result(CONTENT_COUNT);
	std::vector<CONTENT_TYPE> result2(CONTENT_COUNT);
	CUDA_ASSERT(cudaMemcpy(result.data(), result_dev, contentTotalBytes, cudaMemcpyDeviceToHost));
	CUDA_ASSERT(cudaMemcpy(result2.data(), result2_dev, contentTotalBytes, cudaMemcpyDeviceToHost));

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
			CONTENT_TYPE dx2 = std::abs(answer[i] - result2[i]);
			assert(dx == 0);
			assert(dx2 == 0);
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
