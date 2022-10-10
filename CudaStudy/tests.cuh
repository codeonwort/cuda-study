#pragma once

#include <driver_types.h>

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <array>
#include <vector>
#include <algorithm>

extern int runTest_vecAdd(int argc, char* argv[]);
extern int runTest_matMul(int argc, char* argv[]);
extern int runTest_convFilter(int argc, char* argv[]);
extern int runTest_stencil(int argc, char* argv[]);
extern int runTest_histogram(int argc, char* argv[]);
extern int runTest_reduction(int argc, char* argv[]);

inline void CUDA_ASSERT(cudaError_t err) {
	
	if (err != cudaError::cudaSuccess) {
		printf("[ERROR] %s(code=%d): %s\n",
			cudaGetErrorName(err),
			(int)err,
			cudaGetErrorString(err));
	}
}

inline dim3 calcNumBlocks(const dim3& dimensions, const dim3& blockSize) {
	auto iceil = [](int x, int y) { return (x % y) ? x / y + 1 : x / y; };
	int x = iceil(dimensions.x, blockSize.x);
	int y = iceil(dimensions.y, blockSize.y);
	int z = iceil(dimensions.z, blockSize.z);
	return dim3(x, y, z);
}
