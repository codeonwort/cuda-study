// Chapter 10. Reduction

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CONTENT_TYPE  int32_t
#define CONTENT_COUNT 100000

__global__ void kernel_reduction(
	CONTENT_TYPE* content, int32_t totalCount,
	CONTENT_TYPE* outResult)
{
	uint32_t ix = threadIdx.x + (2 * blockDim.x * blockIdx.x);

	if (ix >= totalCount) {
		return;
	}

	for (uint32_t stride = blockDim.x; stride >= 1; stride /= 2) {
		if (threadIdx.x < stride && ix + stride < totalCount) {
			content[ix] += content[ix + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(outResult, content[(2 * blockDim.x * blockIdx.x)]);
	}
}

int runTest_reduction(int argc, char* argv[])
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

	// ------------------------------------------
	// Host -> device

	std::vector<CONTENT_TYPE> content(CONTENT_COUNT, 0);
	for (size_t i = 0; i < CONTENT_COUNT; ++i) {
		content[i] = i + 1;
	}

	const size_t contentTotalBytes = sizeof(CONTENT_TYPE) * CONTENT_COUNT;

	CONTENT_TYPE* content_dev;
	CONTENT_TYPE* result_dev;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&result_dev, sizeof(CONTENT_TYPE)));
	CUDA_ASSERT(cudaMemcpy(content_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemset(result_dev, 0, sizeof(CONTENT_TYPE)));

	// ------------------------------------------
	// Kernel: reduction (sum)

	puts("Run kernel: reduction (sum)");

	const dim3 blockDim(deviceProps.maxThreadsPerBlock, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(CONTENT_COUNT / 2, 1, 1), blockDim);

	kernel_reduction<<<numBlocks, blockDim>>>(
		content_dev, CONTENT_COUNT,
		result_dev);

	// ------------------------------------------
	// Device -> host

	CONTENT_TYPE result;
	CUDA_ASSERT(cudaMemcpy(&result, result_dev, sizeof(CONTENT_TYPE), cudaMemcpyDeviceToHost));

	puts("Compare results...");
	{
		CONTENT_TYPE answer = 0;
		for (CONTENT_TYPE x : content) {
			answer += x;
		}
		assert(answer == result);
	}

	return 0;
}
