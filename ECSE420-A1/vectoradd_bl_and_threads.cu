#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE (2048*2048)
#define THREADS_PER_BLOCK 1024

__global__ void addition(int* a, int* b, int* c, int* n) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < n) {
		c[index] = a[index] + b[index];
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

	// Launch addition() kernel on GPU with SIZE blocks Output:
	addition<<< (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, b, c, SIZE);

	// Wait for GPU threads to complete
	cudaDeviceSynchronize(); 			// ====New Function to sync

	// Printing the Output Array
	printArray(c, 10);

	// Cleanup
	cudaFree(a);cudaFree(b); cudaFree(c);
	
	return 0;
}