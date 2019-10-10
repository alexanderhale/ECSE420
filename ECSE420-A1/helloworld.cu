#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>			  // =========== Handles all the CUDA Syntax
#include <device_launch_parameters.h> // ====== Handles device parameters (threadIdx.x, blockIdx.x)

__global__ void myKernel(void) {
	printf("Hello World!\n");
}

int main(void) {
	myKernel <<<1, 1>>> ();
	return 0;
}