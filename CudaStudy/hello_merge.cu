// Chapter 12. Merge

#include "tests.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CONTENT_TYPE  int32_t
#define INPUT_A_COUNT (1024 * 1024 + 75)
#define INPUT_B_COUNT (1024 * 1024 + 287)

__device__ void merge_sequential(
	CONTENT_TYPE* A, size_t lenA,
	CONTENT_TYPE* B, size_t lenB,
	CONTENT_TYPE* C)
{
	size_t i = 0, j = 0, k = 0;
	while ((i < lenA) && (j < lenB)) {
		if (A[i] <= B[j]) {
			C[k++] = A[i++];
		} else {
			C[k++] = B[j++];
		}
	}
	if (i == lenA) {
		while (j < lenB) {
			C[k++] = B[j++];
		}
	} else {
		while (i < lenA) {
			C[k++] = A[i++];
		}
	}
}

__device__ int32_t co_rank(
	int32_t k,
	CONTENT_TYPE* A, int32_t lenA,
	CONTENT_TYPE* B, int32_t lenB)
{
	int32_t i = k < lenA ? k : lenA;
	int32_t j = k - i;
	int32_t i_low = 0 > (k - lenB) ? 0 : (k - lenB);
	int32_t j_low = 0 > (k - lenA) ? 0 : (k - lenA);
	int32_t delta;
	bool active = true;
	while (active) {
		if (i > 0 && j < lenB && A[i - 1] > B[j]) {
			delta = ((i - i_low + 1) >> 1);
			j_low = j;
			j = j + delta;
			i = i - delta;
		} else if (j > 0 && i < lenA && B[j - 1] >= A[i]) {
			delta = ((j - j_low + 1) >> 1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		} else {
			active = false;
		}
	}
	return i;
}

// Kernel: merge (naive)
// Heavily ALU bound
__global__ void kernel_merge_naive(
	CONTENT_TYPE* A, size_t lenA,
	CONTENT_TYPE* B, size_t lenB,
	CONTENT_TYPE* C)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t elemPerThread = ceilf((float)(lenA + lenB) / (float)(blockDim.x * gridDim.x));
	int32_t k_curr = tid * elemPerThread; // start output index
	if (k_curr >= lenA + lenB) {
		return;
	}
	int32_t k_next = (tid + 1) * elemPerThread; // end output index
	if (k_next > lenA + lenB) {
		k_next = lenA + lenB;
	}
	int32_t i_curr = co_rank(k_curr, A, lenA, B, lenB);
	int32_t i_next = co_rank(k_next, A, lenA, B, lenB);
	int32_t j_curr = k_curr - i_curr;
	int32_t j_next = k_next - i_next;
	merge_sequential(
		A + i_curr, i_next - i_curr,
		B + j_curr, j_next - j_curr,
		C + k_curr);
}

int runTest_merge(int argc, char* argv[])
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

	// ------------------------------------------
	// Host -> device

	std::vector<CONTENT_TYPE> inputA(INPUT_A_COUNT);
	std::vector<CONTENT_TYPE> inputB(INPUT_B_COUNT);
	std::vector<CONTENT_TYPE> answer(INPUT_A_COUNT + INPUT_B_COUNT);

	// Prepare input
	srand(time(NULL));
	for (size_t i = 0; i < INPUT_A_COUNT; ++i)
	{
		inputA[i] = rand() - (RAND_MAX / 2);
		inputA[i] *= rand() - (RAND_MAX / 2);
	}
	for (size_t i = 0; i < INPUT_B_COUNT; ++i)
	{
		inputB[i] = rand() - (RAND_MAX / 2);
		inputB[i] *= rand() - (RAND_MAX / 2);
	}
	std::sort(inputA.begin(), inputA.end());
	std::sort(inputB.begin(), inputB.end());

	// Find and verify answer
	for (size_t i = 0; i < INPUT_A_COUNT; ++i) {
		answer[i] = inputA[i];
	}
	for (size_t i = INPUT_A_COUNT; i < answer.size(); ++i) {
		answer[i] = inputB[i - INPUT_A_COUNT];
	}
	std::sort(answer.begin(), answer.end());
	for (size_t i = 1; i < answer.size(); ++i) {
		assert(answer[i - 1] <= answer[i]);
	}

	printf("inputA: %zu elements\n", inputA.size());
	printf("inputB: %zu elements\n", inputB.size());

	// ------------------------------------------
	// Kernel: merge

	const size_t inputA_totalBytes = sizeof(CONTENT_TYPE) * INPUT_A_COUNT;
	const size_t inputB_totalBytes = sizeof(CONTENT_TYPE) * INPUT_B_COUNT;
	const size_t result_totalBytes = inputA_totalBytes + inputB_totalBytes;

	CONTENT_TYPE* inputA_dev;
	CONTENT_TYPE* inputB_dev;
	CONTENT_TYPE* result_dev;
	CUDA_ASSERT(cudaMalloc(&inputA_dev, inputA_totalBytes));
	CUDA_ASSERT(cudaMalloc(&inputB_dev, inputB_totalBytes));
	CUDA_ASSERT(cudaMalloc(&result_dev, result_totalBytes));
	CUDA_ASSERT(cudaMemcpy(inputA_dev, inputA.data(), inputA_totalBytes, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(inputB_dev, inputB.data(), inputB_totalBytes, cudaMemcpyHostToDevice));

	puts("Kernel: merge");
	{
		const dim3 blockSize(deviceProps.maxThreadsPerBlock, 1, 1);
		const dim3 totalSize(INPUT_A_COUNT + INPUT_B_COUNT, 1, 1);
		const dim3 numBlocks = calcNumBlocks(totalSize, blockSize);
		kernel_merge_naive<<<numBlocks, blockSize>>>(
			inputA_dev, INPUT_A_COUNT,
			inputB_dev, INPUT_B_COUNT,
			result_dev);
	}

	// ------------------------------------------
	// Device -> host

	std::vector<CONTENT_TYPE> result(INPUT_A_COUNT + INPUT_B_COUNT);
	CUDA_ASSERT(cudaMemcpy(result.data(), result_dev, result_totalBytes, cudaMemcpyDeviceToHost));

	puts("Compare results...");
	for (size_t i = 0; i < result.size(); ++i) {
		assert(answer[i] == result[i]);
	}

	return 0;
}
