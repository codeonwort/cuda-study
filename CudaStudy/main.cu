#include "tests.cuh"
#include <stdio.h>

void printTestHeader(const char* header) {
	puts("");
	printf("=== TEST: %s ===\n", header);
	puts("");
}

int main(int argc, char* argv[]) {
	//printTestHeader("VECTOR ADD");
	//runTest_vecAdd(argc, argv);

	//printTestHeader("MATRIX MULTIPLICATION");
	//runTest_matMul(argc, argv);

	//printTestHeader("CONVOLUTION FILTER");
	//runTest_convFilter(argc, argv);

	//printTestHeader("STENCIL");
	//runTest_stencil(argc, argv);

	//printTestHeader("HISTOGRAM");
	//runTest_histogram(argc, argv);

	//printTestHeader("REDUCTION");
	//runTest_reduction(argc, argv);

	//printTestHeader("PREFIX SUM (SCAN)");
	//runTest_scan(argc, argv);

	printTestHeader("MERGE");
	runTest_merge(argc, argv);

	return 0;
}
