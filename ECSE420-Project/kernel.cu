
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Defined min and max functions 
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


__global__ void mergeSort(int* startingArr, int* endingArr, unsigned int N, unsigned int level, unsigned int threads) {

	// Assigns thread ID 
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Creates individual portions to be sorted based off level, adds 1 if a remainder exists 
	unsigned int totalArrays = N / (level);
	if (totalArrays * level != N) {
		totalArrays = totalArrays + 1;
	}

	// Initial start is based off of ID and current level 
	unsigned arrayBeginning = id * level;

	// For loop for mergesort, increments by number of threads over time 
	for (unsigned int currentArray = id; currentArray < totalArrays; currentArray = currentArray + threads) {

		// Takes middle and end values for single sort
		unsigned arrayMiddle = MIN(arrayBeginning + (level / 2), N);
		unsigned arrayEnd = MIN(arrayBeginning + level, N);

		// Creates indexes for individual sort to increment
		unsigned int leftArrayIndex = arrayBeginning;
		unsigned int rightArrayIndex = arrayMiddle;

		// Individual merge for given "level" worth of values 
		for (unsigned int i = arrayBeginning; i < arrayEnd; i++) {

			// If middle array is larger, temp start is index in sorted array 
			if (leftArrayIndex < arrayMiddle && (rightArrayIndex >= arrayEnd || startingArr[leftArrayIndex] < startingArr[rightArrayIndex])) {
				endingArr[i] = startingArr[leftArrayIndex];
				leftArrayIndex++;
			}

			// If middle array start is smaller, middle array is index in sorted array 
			else {

				endingArr[i] = startingArr[rightArrayIndex];
				rightArrayIndex++;
			}
		}

		// Increments to next start point based off of thread count 
		arrayBeginning = arrayBeginning + threads * level;

	}

}

int main() {

	// Define
	printf("Parallelized Merge Sort Has Begun");

	unsigned int N = 1000000;
	unsigned int threads = 32;

	// Define arrays for system
	int* startingArray = new int[N];
	int* endingArray = new int[N];
	int* cudaStartArray;
	int* cudaEndArray;

	// Initialize starting array to values of 0 to 10000 randomly   
	for (int i = 0; i < N; i++) {
		startingArray[i] = rand() % 10000;
	}

	// Copy starting Array values into end Array

	memcpy(endingArray, startingArray, N * sizeof(int));

	// Clock values 

	clock_t clockStart, clockEnd, clockTemp;
	clockStart = clock();

	for (unsigned int level = 2; level < N * 2; level = level * 2) {

		cudaSetDevice(0);

		// Allocate cuda memory for start and end array 

		cudaMalloc((void**)& cudaStartArray, N * sizeof(int));
		cudaMalloc((void**)& cudaEndArray, N * sizeof(int));

		// Copy data from original arrays into cuda version 

		cudaMemcpy(cudaStartArray, startingArray, N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaEndArray, endingArray, N * sizeof(int), cudaMemcpyHostToDevice);

		// Change thread count based off of input 

		if (threads > 1024) {
			threads = 1024;
		}

		// Merge sort formula 

		mergeSort << <1, threads >> > (cudaStartArray, cudaEndArray, N, level, threads);

		// Synchronize threads 

		cudaDeviceSynchronize();

		// Copy end array outcome from cuda version to original

		cudaMemcpy(endingArray, cudaEndArray, N * sizeof(int), cudaMemcpyDeviceToHost);

		// Free cuda memory 

		cudaFree(cudaStartArray);
		cudaFree(cudaEndArray);
		clockTemp = clock();
		printf("\nAt Iteration : %d || Max Sections Sorted : %d  || Iteration Sort Time :  %f", (int)log2(level), level, ((float)(clockTemp - clockStart)) / CLOCKS_PER_SEC);

		/*
		//Prints individual value at each index
		for (int j = 0; j < N; j++) {
			printf("%d ", endingArray[j]);
			printf("\n");
		}
		*/

		// Copies endingArray to startingArray  
		memcpy(startingArray, endingArray, N * sizeof(int));

	}

	// End time clock value  
	clockEnd = clock();

	printf(" \nFINAL ITERATION : ");

	// Checks to see if merge sorted matrix is properly sorted 
	bool isTrue = true;
	for (unsigned int i = 0; i < N; i++) {
		//printf("%d \n", endingArray[j]);
		if (i > 0) {
			if (endingArray[i] < endingArray[i - 1]) {
				isTrue = false;
			}
		}
	}

	// Prints if matrix has or hasn't been sorted 
	if (isTrue) {
		printf("The Input Array Is Sorted From Merge Sort \n");
	}
	else {
		printf("The Input Array Is Not Sorted From Merge Sort \n");
	}

	// Prints total time of system 
	printf("Total Merge Sort Time :  %f", ((float)(clockEnd - clockStart)) / CLOCKS_PER_SEC);

	return 0;
}


