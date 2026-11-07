#include "tests.cuh"
#include <stdio.h>

using TestFn = int (*)(int argc, char* argv[]);

struct TestCase {
	const char* name;
	TestFn pfn;
};
TestCase testCases[] = {
	{ "VECTOR ADD", runTest_vecAdd },
	{ "MATRIX MULTIPLICATION", runTest_matMul },
	{ "CONVOLUTION FILTER", runTest_convFilter },
	{ "STENCIL", runTest_stencil },
	{ "HISTOGRAM", runTest_histogram },
	{ "REDUCTION", runTest_reduction },
	{ "PREFIX SUM (SCAN)", runTest_scan },
	//{ "MERGE", runTest_merge }, // TODO: Finish merge kernel
};
int kNumCases = _countof(testCases);

void printTestHeader(const char* header) {
	puts("");
	printf_s("=== TEST: %s ===\n", header);
	puts("");
}

int main(int argc, char* argv[]) {
	int number = -1;
	if (argc >= 2)
	{
		scanf_s("%d", &number);
	}
	else
	{
		for (int i = 0; i < kNumCases; ++i)
		{
			printf_s("%d - %s\n", i, testCases[i].name);
		}
	}

	if (number == -1)
	{
		printf_s("Enter test number: ");
		scanf_s("%d", &number);
	}

	if (0 <= number && number < kNumCases)
	{
		printTestHeader(testCases[number].name);
		testCases[number].pfn(argc, argv);
	}
	else
	{
		printf_s("Invalid test number: %d", number);
		return 1;
	}

	return 0;
}
