// Chapter 10. Reduction

// Some metrics by Nsight Compute (on RTX 3080 Ti)
//                          naive      sharedMem   coarsening(4)   coarsening(16)
// Cycles                 : 17309592   18166517    7779121         4459195
// Duration (ms)          : 13.10      13.73       6.15            3.27
// Compute Throughput (%) : 36.15      34.90       47.68           55.62
// Memory Throughput (%)  : 5.57       5.19        8.30            14.27
// Register/thread	      : 24         24          26              26
//
// -> sharedMem is actually slower than naive?

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CONTENT_TYPE  int32_t
#define CONTENT_COUNT 100000000
//#define CONTENT_COUNT 1024
#define COARSENING    16

__global__ void kernel_reduction_naive(
	CONTENT_TYPE* content, int32_t totalCount,
	CONTENT_TYPE* outResult)
{
	uint32_t ix = threadIdx.x + (2 * blockDim.x * blockIdx.x);

	if (ix >= totalCount) {
		return;
	}

	for (uint32_t stride = blockDim.x; stride >= 1; stride /= 2) {
		if (threadIdx.x < stride && ix + stride < totalCount) {
			content[ix] = content[ix] ^ content[ix + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicXor(outResult, content[(2 * blockDim.x * blockIdx.x)]);
	}
}

__global__ void kernel_reduction_sharedMem(
	CONTENT_TYPE* content, int32_t totalCount,
	CONTENT_TYPE* outResult)
{
	uint32_t ix = threadIdx.x + (2 * blockDim.x * blockIdx.x);

	// Assumes blockDim.x = 1024
	__shared__ CONTENT_TYPE content_s[1024];

	if (ix >= totalCount) {
		return;
	}

	if (ix + blockDim.x < totalCount) {
		content_s[threadIdx.x] = content[ix] ^ content[ix + blockDim.x];
	} else {
		content_s[threadIdx.x] = content[ix];
	}

	for (uint32_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (threadIdx.x < stride && ix + stride < totalCount) {
			content_s[threadIdx.x] = content_s[threadIdx.x] ^ content_s[threadIdx.x + stride];
		}
	}
	if (threadIdx.x == 0) {
		atomicXor(outResult, content_s[0]);
	}
}

__global__ void kernel_reduction_coarsening(
	CONTENT_TYPE* content, int32_t totalCount,
	CONTENT_TYPE* outResult)
{
	uint32_t ix = threadIdx.x + (2 * COARSENING * blockDim.x * blockIdx.x);

	// Assumes blockDim.x = 1024
	__shared__ CONTENT_TYPE content_s[1024];
	content_s[threadIdx.x] = 0;
	__syncthreads();

	if (ix >= totalCount) {
		return;
	}

	for (size_t p = 0; p < COARSENING; ++p) {
		size_t q = ix + p * blockDim.x;
		if (q < totalCount) {
			content_s[threadIdx.x] = content_s[threadIdx.x] ^ content[q];
		}
	}
	for (size_t p = 0; p < COARSENING; ++p) {
		size_t q = ix + (p + COARSENING) * blockDim.x;
		if (q < totalCount) {
			content_s[threadIdx.x] = content_s[threadIdx.x] ^ content[q];
		}
	}
	for (uint32_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (threadIdx.x < stride && ix + stride < totalCount) {
			content_s[threadIdx.x] = content_s[threadIdx.x] ^ content_s[threadIdx.x + stride];
		}
	}
	if (threadIdx.x == 0) {
		atomicXor(outResult, content_s[0]);
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
	srand(time(NULL));
	for (size_t i = 0; i < CONTENT_COUNT; ++i) {
		content[i] = (rand() * rand()) % (int32_t)1e8;
	}
	printf("Find xor of %d elements\n", CONTENT_COUNT);

	const size_t contentTotalBytes = sizeof(CONTENT_TYPE) * CONTENT_COUNT;

	CONTENT_TYPE* content_dev;
	CONTENT_TYPE* content2_dev;
	CONTENT_TYPE* content3_dev;
	CONTENT_TYPE* result_dev;
	CONTENT_TYPE* result2_dev;
	CONTENT_TYPE* result3_dev;
	//CONTENT_TYPE result_initial = std::numeric_limits<CONTENT_TYPE>::lowest();
	CONTENT_TYPE result_initial = 0;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&content2_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&content3_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&result_dev, sizeof(CONTENT_TYPE)));
	CUDA_ASSERT(cudaMalloc(&result2_dev, sizeof(CONTENT_TYPE)));
	CUDA_ASSERT(cudaMalloc(&result3_dev, sizeof(CONTENT_TYPE)));
	CUDA_ASSERT(cudaMemcpy(content_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(content2_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(content3_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(result_dev, &result_initial, sizeof(CONTENT_TYPE), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(result2_dev, &result_initial, sizeof(CONTENT_TYPE), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(result3_dev, &result_initial, sizeof(CONTENT_TYPE), cudaMemcpyHostToDevice));

	// ------------------------------------------
	// Kernel: reduction (xor)

	puts("Run kernel: reduction (xor)");

	const dim3 blockDim(deviceProps.maxThreadsPerBlock, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(CONTENT_COUNT / 2, 1, 1), blockDim);

	kernel_reduction_naive<<<numBlocks, blockDim>>>(
		content_dev, CONTENT_COUNT,
		result_dev);

	kernel_reduction_sharedMem<<<numBlocks, blockDim>>>(
		content2_dev, CONTENT_COUNT,
		result2_dev);

	const dim3 numBlocksCoarsed
		= calcNumBlocks(dim3(CONTENT_COUNT / (2 * COARSENING), 1, 1), blockDim);
	kernel_reduction_coarsening<<<numBlocksCoarsed, blockDim>>>(
		content3_dev, CONTENT_COUNT,
		result3_dev);

	// ------------------------------------------
	// Device -> host

	CONTENT_TYPE result, result2, result3;
	CUDA_ASSERT(cudaMemcpy(&result, result_dev, sizeof(CONTENT_TYPE), cudaMemcpyDeviceToHost));
	CUDA_ASSERT(cudaMemcpy(&result2, result2_dev, sizeof(CONTENT_TYPE), cudaMemcpyDeviceToHost));
	CUDA_ASSERT(cudaMemcpy(&result3, result3_dev, sizeof(CONTENT_TYPE), cudaMemcpyDeviceToHost));

	puts("Compare results...");
	{
		//CONTENT_TYPE answer = std::numeric_limits<CONTENT_TYPE>::lowest();
		CONTENT_TYPE answer = 0;
		for (CONTENT_TYPE x : content) {
			answer = answer ^ x;
		}
		assert(answer == result);
		assert(answer == result2);
		assert(answer == result3);
		printf("XOR: %d\n", answer);
	}

	return 0;
}
