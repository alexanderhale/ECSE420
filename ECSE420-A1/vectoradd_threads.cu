#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 1024

__global__ void addition(int* a, int* b, int* c, int n) {
	/*for (int i = 0; i < n; i++) { // Conventional way of Vector Addition with single thread
	c[i] = a[i] + b[i];
	}*/

	int i = threadIdx.x; 			// Using GPU Blocks for Vector Addition
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

// Printing an Array
void printArray(int* a, int n) {
	for (int i = 0; i < n; i++) {
		printf("c[%d] = %d\n", i, a[i]);
	}
}

int main(void) {
	int* a, * b, * c;
	// Allocate space in Unified Memory for a,b,c
	cudaMallocManaged((void**)&a, SIZE * sizeof(int)); 	// Different than cudaMalloc()
	cudaMallocManaged((void**)&b, SIZE * sizeof(int));
	cudaMallocManaged((void**)&c, SIZE * sizeof(int));

	// Initializing the values
	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// Launch addition() kernel on GPU with SIZE threads
	addition<<< 1, SIZE >>>(a, b, c, SIZE);

	// Wait for GPU threads to complete
	cudaDeviceSynchronize(); 			// ====New Function to sync

	// Printing the Output Array
	printArray(c, 10);

	// Cleanup
	cudaFree(a);cudaFree(b); cudaFree(c);
	
	return 0;
}