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

	printTestHeader("CONVOLUTION FILTER");
	runTest_convFilter(argc, argv);

	return 0;
}
