// Chapter 9. Parallel histogram

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <array>
#include <algorithm>

// A to Z
#define NUM_CATEGORIES 26

// Naive version that uses atomic operations across all threads.
__global__ void kernel_histogram_naive(
	uint8_t* content, size_t totalLength,
	uint32_t* outHistogram)
{
	int ix = threadIdx.x + (blockDim.x * blockIdx.x);
	if (ix >= totalLength) {
		return;
	}

	int cat = int(content[ix]) - 'a';
	if (0 <= cat && cat < NUM_CATEGORIES) {
		atomicAdd(outHistogram + cat, 1);
	}
}

// Privatization: Contention occurs only between threads in the same block and merging step.
__global__ void kernel_histogram_private(
	uint8_t* content, size_t totalLength,
	uint32_t* outHistogram)
{
	__shared__ uint32_t histo_s[NUM_CATEGORIES];
	if (threadIdx.x < NUM_CATEGORIES) {
		histo_s[threadIdx.x] = 0;
	}
	__syncthreads();

	int ix = threadIdx.x + (blockDim.x * blockIdx.x);
	if (ix >= totalLength) {
		return;
	}

	int cat = int(content[ix]) - 'a';
	if (0 <= cat && cat < NUM_CATEGORIES) {
		atomicAdd(histo_s + cat, 1);
	}
	__syncthreads();

	// Assumes blockDim.x >= NUM_CATEGORIES
	if (threadIdx.x < NUM_CATEGORIES) {
		if (histo_s[threadIdx.x] > 0) {
			atomicAdd(outHistogram + threadIdx.x, histo_s[threadIdx.x]);
		}
	}
}

int runTest_histogram(int argc, char** argv)
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
	// Read content

	const char* filename = "content/vulkan_handles.hpp";

	printf("Open content file: %s\n", filename);

	FILE* file = fopen(filename, "r");
	if (file == nullptr) {
		printf("Can't find: %s\n", filename);
		puts("Check if current directory is the solution dir");
		return 1;
	}

	fseek(file, 0, SEEK_END);
	size_t fileSize = (size_t)ftell(file);
	fseek(file, 0, SEEK_SET);

	std::vector<uint8_t> content(fileSize, 0);
	fread_s(content.data(), content.size(), sizeof(uint8_t), fileSize, file);

	printf("Content read: %zu bytes\n", fileSize);

	fclose(file);

	// ------------------------------------------
	// Host -> device

	const size_t contentLength = content.size();
	const size_t contentTotalBytes = sizeof(uint8_t) * content.size();
	const size_t categoryTotalBytes = sizeof(uint32_t) * NUM_CATEGORIES;

	uint8_t* content_dev;
	uint32_t* histogram_dev;
	uint32_t* histogram2_dev;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram2_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMemcpy(content_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemset(histogram_dev, 0, categoryTotalBytes));
	CUDA_ASSERT(cudaMemset(histogram2_dev, 0, categoryTotalBytes));

	// ------------------------------------------
	// Kernel: histogram

	puts("Run kernel: naive histogram");

	const dim3 blockDim(deviceProps.maxThreadsPerBlock, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(contentLength, 1, 1), blockDim);

	printf("\tblockDim: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
	printf("\tnumBlocks: (%u, %u, %u)\n", numBlocks.x, numBlocks.y, numBlocks.z);

	kernel_histogram_naive<<<numBlocks, blockDim>>>(
		content_dev, contentLength,
		histogram_dev);

	kernel_histogram_private<<<numBlocks, blockDim>>>(
		content_dev, contentLength,
		histogram2_dev);

	// ------------------------------------------
	// Device -> host

	std::array<uint32_t, NUM_CATEGORIES> histogram;
	cudaMemcpy(histogram.data(), histogram_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	std::array<uint32_t, NUM_CATEGORIES> histogram2;
	cudaMemcpy(histogram2.data(), histogram2_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	puts("Compare results...");
	{
		for (size_t i = 0; i < histogram.size(); ++i) {
			assert(histogram[i] == histogram2[i]);
		}
	}

	uint32_t maxCount = 0;
	for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
		printf("(%c, %u) ", (unsigned char)('a' + i), histogram[i]);
		maxCount = std::max(maxCount, histogram[i]);
	}
	puts("");

	for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
		printf("%c", (unsigned char)('a' + i));
	}
	puts("");

	const size_t numRows = 10;
	for (size_t row = 0; row < numRows; ++row) {
		for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
			float ratio = (float)histogram[i] / (float)maxCount;
			if (size_t(ratio * numRows) >= row) {
				printf("*");
			} else {
				printf(" ");
			}
		}
		puts("");
	}

	return 0;
}
