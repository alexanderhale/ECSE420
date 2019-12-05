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


__global__ void mergeSort(int* startingArr, int* endingArr, unsigned int N, unsigned int sortedArraySize, unsigned int threads, unsigned int totalSortedArrays) {

	// Assigns thread ID 
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

	// index of the first element of the first sorted array that this thread is responsible for
	unsigned startElementIndex = id * sortedArraySize;

	// loop through each of the sorted arrays that this thread is responsible for merging
	for (unsigned int currentArray = id; currentArray < totalSortedArrays; currentArray += threads) {

		// calculate the middle and end element indices for this merge
		unsigned arrayMiddle = MIN(startElementIndex + (sortedArraySize / 2), N);
		unsigned arrayEnd = MIN(startElementIndex + sortedArraySize, N);

		// create variables to hold the left and right indices of the merge as it proceeds
		unsigned int leftArrayIndex = startElementIndex;
		unsigned int rightArrayIndex = arrayMiddle;

		// perform the merge of two sorted sub-arrays into one sorted array
		for (unsigned int i = startElementIndex; i < arrayEnd; i++) {

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

		// increment to the start point of the next merge
		startElementIndex += threads * sortedArraySize;
	}
}

int main() {
	printf("Parallelized Merge Sort Has Begun");

	unsigned int N = 1000000;
	unsigned int threads = 32;
	if (threads > 1024) {
		threads = 1024;
	}

	// Define arrays for system
	int* startingArray = new int[N];
	int* endingArray = new int[N];
	int* cudaStartArray;
	int* cudaEndArray;

	// Initialize array to random values 0 and 10000   
	for (int i = 0; i < N; i++) {
		startingArray[i] = rand() % 10000;
	}

	// Copy starting Array values into end Array
	memcpy(endingArray, startingArray, N * sizeof(int));

	// Clock variable
	clock_t startTime = clock();

	// iterate through the levels of the merge tree, using the size of the individual sorted arrays at each level as the loop index
	for (unsigned int sortedArraySize = 2; sortedArraySize < N * 2; sortedArraySize = sortedArraySize * 2) {
		clock_t loopTime = clock();

		cudaSetDevice(0);

		// Allocate memory on device for the original list and the list at the end of this step
		cudaMalloc((void**)& cudaStartArray, N * sizeof(int));
		cudaMalloc((void**)& cudaEndArray, N * sizeof(int));

		// Copy data from host to device
		cudaMemcpy(cudaStartArray, startingArray, N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaEndArray, endingArray, N * sizeof(int), cudaMemcpyHostToDevice);

		// calculate how many individual arrays there are at this level of the merge tree
		unsigned int totalSortedArrays = N / sortedArraySize + (N % sortedArraySize != 0);

		// call MergeSort kernel function
		mergeSort <<<1, threads>>> (cudaStartArray, cudaEndArray, N, sortedArraySize, threads, totalSortedArrays);

		// Synchronize threads 
		cudaDeviceSynchronize();

		// Copy resulting list to host
		cudaMemcpy(endingArray, cudaEndArray, N * sizeof(int), cudaMemcpyDeviceToHost);

		// Free cuda memory 
		cudaFree(cudaStartArray);
		cudaFree(cudaEndArray);

		// Copies endingArray to startingArray to prepare for the next step
		memcpy(startingArray, endingArray, N * sizeof(int));

		// printing timing results
		printf("\nMerge Tree Level: %d || Size of Sorted Arrays: %d  || Level Merge Time:  %f", (int)log2(sortedArraySize), sortedArraySize, (float)(clock() - loopTime) / CLOCKS_PER_SEC);
	}

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
	printf("Total Merge Sort Time :  %f", ((float)(clock() - startTime)) / CLOCKS_PER_SEC);

	return 0;
}