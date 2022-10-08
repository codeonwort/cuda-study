#pragma once

#include <driver_types.h>

#include <assert.h>
#include <stdio.h>

extern int runTest_vecAdd(int argc, char* argv[]);
extern int runTest_matMul(int argc, char* argv[]);
extern int runTest_convFilter(int argc, char* argv[]);
extern int runTest_stencil(int argc, char* argv[]);

inline void CUDA_ASSERT(cudaError_t err) {
	
	if (err != cudaError::cudaSuccess) {
		printf("[ERROR] %s(code=%d): %s\n",
			cudaGetErrorName(err),
			(int)err,
			cudaGetErrorString(err));
	}
}
