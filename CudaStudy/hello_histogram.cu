// Chapter 9. Parallel histogram

// Some metrics by Nsight Compute (on RTX 3080 Ti)
//                          naive    private   coarsening   aggregate
// Cycles                 : 207627   92679     81954        93840
// Duration (micro)       : 152.99   69.15     61.44        69.73
// Compute Throughput (%) : 9.78     25.14     35.05        39.52
// Memory Throughput (%)  : 6.65     11.06     12.90        10.52
// Register/thread	      : 24       26        30           34

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
#define COARSENING     4

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

// Thread coarsening: Each thread processes multiple elements.
__global__ void kernel_histogram_coarsening(
	uint8_t* content, size_t totalLength,
	uint32_t* outHistogram)
{
	__shared__ uint32_t histo_s[NUM_CATEGORIES];
	if (threadIdx.x < NUM_CATEGORIES) {
		histo_s[threadIdx.x] = 0;
	}
	__syncthreads();

	for (int phase = 0; phase < COARSENING; ++phase) {
		int ix = threadIdx.x + (phase * blockDim.x) + (COARSENING * blockDim.x * blockIdx.x);
		if (ix >= totalLength) {
			break;
		}

		int cat = int(content[ix]) - 'a';
		if (0 <= cat && cat < NUM_CATEGORIES) {
			atomicAdd(histo_s + cat, 1);
		}
	}
	__syncthreads();

	// Assumes blockDim.x >= NUM_CATEGORIES
	if (threadIdx.x < NUM_CATEGORIES) {
		if (histo_s[threadIdx.x] > 0) {
			atomicAdd(outHistogram + threadIdx.x, histo_s[threadIdx.x]);
		}
	}
}

// Aggregation: Group consecutive updates to the same element.
// NOTE: This is actually slower than non-aggregation version,
//       which implies the contention rate of 'content' is low.
__global__ void kernel_histogram_aggregate(
	uint8_t* content, size_t totalLength,
	uint32_t* outHistogram)
{
	__shared__ uint32_t histo_s[NUM_CATEGORIES];
	if (threadIdx.x < NUM_CATEGORIES) {
		histo_s[threadIdx.x] = 0;
	}
	__syncthreads();

	int prevCat = -1;
	int accum = 0;
	for (int phase = 0; phase < COARSENING; ++phase) {
		int ix = threadIdx.x + (phase * blockDim.x) + (COARSENING * blockDim.x * blockIdx.x);
		if (ix >= totalLength) {
			break;
		}

		int cat = int(content[ix]) - 'a';
		if (0 <= cat && cat < NUM_CATEGORIES) {
			if (cat == prevCat) {
				++accum;
			} else {
				if (accum > 0) {
					atomicAdd(histo_s + prevCat, accum);
				}
				accum = 1;
				prevCat = cat;
			}
		}
	}
	if (accum > 0) {
		atomicAdd(histo_s + prevCat, accum);
	}
	__syncthreads();

	// Assumes blockDim.x >= NUM_CATEGORIES
	if (threadIdx.x < NUM_CATEGORIES) {
		if (histo_s[threadIdx.x] > 0) {
			atomicAdd(outHistogram + threadIdx.x, histo_s[threadIdx.x]);
		}
	}
}

int runTest_histogram(int argc, char* argv[])
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
	uint32_t* histogram3_dev;
	uint32_t* histogram4_dev;
	CUDA_ASSERT(cudaMalloc(&content_dev, contentTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram2_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram3_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMalloc(&histogram4_dev, categoryTotalBytes));
	CUDA_ASSERT(cudaMemcpy(content_dev, content.data(), contentTotalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemset(histogram_dev, 0, categoryTotalBytes));
	CUDA_ASSERT(cudaMemset(histogram2_dev, 0, categoryTotalBytes));
	CUDA_ASSERT(cudaMemset(histogram3_dev, 0, categoryTotalBytes));
	CUDA_ASSERT(cudaMemset(histogram4_dev, 0, categoryTotalBytes));

	// ------------------------------------------
	// Kernel: histogram

	puts("Run kernel: naive histogram");

	const dim3 blockDim(deviceProps.maxThreadsPerBlock, 1, 1);
	const dim3 numBlocks = calcNumBlocks(dim3(contentLength, 1, 1), blockDim);

	// NOTE: kernel_histogram_coarsening should use 'blockDim'.
	// 'blockDimCoarsed' is here just to calculate 'numBlocksCoarsed'.
	const dim3 blockDimCoarsed(COARSENING * deviceProps.maxThreadsPerBlock, 1, 1);
	const dim3 numBlocksCoarsed = calcNumBlocks(dim3(contentLength, 1, 1), blockDim);

	printf("\tblockDim: (%u, %u, %u)\n", blockDim.x, blockDim.y, blockDim.z);
	printf("\tnumBlocks: (%u, %u, %u)\n", numBlocks.x, numBlocks.y, numBlocks.z);

	kernel_histogram_naive<<<numBlocks, blockDim>>>(
		content_dev, contentLength,
		histogram_dev);

	kernel_histogram_private<<<numBlocks, blockDim>>>(
		content_dev, contentLength,
		histogram2_dev);

	kernel_histogram_coarsening<<<numBlocksCoarsed, blockDim>>>(
		content_dev, contentLength,
		histogram3_dev);

	kernel_histogram_aggregate<<<numBlocksCoarsed, blockDim>>>(
		content_dev, contentLength,
		histogram4_dev);

	// ------------------------------------------
	// Device -> host

	std::array<uint32_t, NUM_CATEGORIES> histogram;
	cudaMemcpy(histogram.data(), histogram_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	std::array<uint32_t, NUM_CATEGORIES> histogram2;
	cudaMemcpy(histogram2.data(), histogram2_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	std::array<uint32_t, NUM_CATEGORIES> histogram3;
	cudaMemcpy(histogram3.data(), histogram3_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	std::array<uint32_t, NUM_CATEGORIES> histogram4;
	cudaMemcpy(histogram4.data(), histogram4_dev, categoryTotalBytes, cudaMemcpyDeviceToHost);

	puts("Compare results...");
	{
		for (size_t i = 0; i < histogram.size(); ++i) {
			assert(histogram[i] == histogram2[i]);
			assert(histogram[i] == histogram3[i]);
			assert(histogram[i] == histogram4[i]);
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
