
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_MSE 0.00001f

__global__ void rectification(unsigned char* image, unsigned char* new_image, unsigned int size) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < size) {
		if (image[index] < 127) {
			new_image[index] = 127;
		}
		else {
			new_image[index] = image[index];
		}
	}
}

__global__ void compression(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int size, unsigned int blocks_per_row) {
	unsigned int index = threadIdx.x + (blockIdx.x % blocks_per_row) * blockDim.x;
	unsigned int new_index = (threadIdx.x + blockIdx.x * blockDim.x) + 4;

	if (index < size) {
		for (int i = 0; i < 4; i++) {						// iterate through R, G, B, A
			unsigned int max = image[index];
			if (image[index + 4 + i] > max) {				// pixel to the right
				max = image[index + 4 + i];
			}
			if (image[index + (4 * width) + i] > max) {		// pixel below
				max = image[index + (4 * width) + i];
			}
			if (image[index + (4 * width) + 4 + i] > max) {	// pixel below & to the right
				max = image[index + (4 * width) + 4 + i];
			}
			new_image[new_index + i] = max;
		}
	}
}

float get_MSE(char* input_filename_1, char* input_filename_2)
{
	unsigned error1, error2;
	unsigned char* image1, * image2;
	unsigned width1, height1, width2, height2;

	error1 = lodepng_decode32_file(&image1, &width1, &height1, input_filename_1);
	error2 = lodepng_decode32_file(&image2, &width2, &height2, input_filename_2);
	if (error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));
	if (error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));
	if (width1 != width2) printf("images do not have same width\n");
	if (height1 != height2) printf("images do not have same height\n");

	// process image
	float im1, im2, diff, sum, MSE;
	sum = 0;
	for (int i = 0; i < width1 * height1; i++) {
		im1 = (float)image1[i];
		im2 = (float)image2[i];
		diff = im1 - im2;
		sum += diff * diff;
	}
	MSE = sqrt(sum) / (width1 * height1);

	free(image1);
	free(image2);

	return MSE;
}

int main()
{
	// file definitions
	char* filename1 = "test.png";						// change these depending on what we're doing
	char* filename2 = "test_rectify_result.png";		// output for rectify
	char* filename3 = "test_pooling_result.png";		// output for pooling
	char* filename4 = "test_rectify_expected_result.png";	// filename for rectify comparison
	char* filename5 = "test_pooling_expected_result.png";	// filename for pooling comparison

	// load input image
	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, filename1);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	unsigned int size_image = width * height * 4 * sizeof(unsigned char); // height x width number of pixels, 4 layers (RGBA) for each pixel, 1 char for each value

	// define number of threads
	unsigned int thread_number = 1000;		// number of threads per block we're using
	unsigned int thread_max = 1024;			// hardware limit: maximum number of threads per block

	if (thread_number > thread_max) {		// can't have more threads than the hardware limit
		thread_number = thread_max;
	}

	/********** Rectify Start ***********/
	// allocate memory space on GPU
	unsigned char* cuda_image, * cuda_new_image;
	cudaMalloc((void**)& cuda_image, size_image);
	cudaMalloc((void**)& cuda_new_image, size_image);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_image, image, size_image, cudaMemcpyHostToDevice);

	// figure out how many blocks we need for this task
	unsigned int num_blocks = (size_image + thread_number - 1) / thread_number;

	// call method on GPU
	rectification <<< num_blocks, thread_number >>> (cuda_image, cuda_new_image, size_image);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned char* new_image = (unsigned char*)malloc(size_image);
	cudaMemcpy(new_image, cuda_new_image, size_image, cudaMemcpyDeviceToHost);
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

	lodepng_encode32_file(filename2, new_image, width, height);
	/********** Rectify End ***********/

	/********** Pooling Start ***********/
	// allocate memory space on GPU
	unsigned char* cuda_image_pool, * cuda_new_image_pool;
	cudaMalloc((void**)& cuda_image_pool, size_image);
	cudaMalloc((void**)& cuda_new_image_pool, size_image);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_image_pool, image, size_image, cudaMemcpyHostToDevice);

	// maximum number of threads we can use is 1 per 16 pixels
		// that's because we can use maximum 1 thread per 2x2 area, and each pixel in that 2x2 area has 4 values
	if (thread_number > ceil(size_image / 16)) {
		thread_number = ceil(size_image / 16);
	}

	// figure out how many blocks we need for this task
	num_blocks = ceil((size_image / thread_number) / 16) + 1;
	unsigned int blocks_per_row = ceil(width / thread_number);

	// call method on GPU
	compression <<< num_blocks, thread_number >>> (cuda_image_pool, cuda_new_image_pool, width, size_image, blocks_per_row);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned char* new_image_pool = (unsigned char*)malloc(size_image);
	cudaMemcpy(new_image_pool, cuda_new_image_pool, size_image, cudaMemcpyDeviceToHost);
	cudaFree(cuda_image_pool);
	cudaFree(cuda_new_image_pool);

	lodepng_encode32_file(filename3, new_image_pool, width / 2, height / 2);
	/********** Pooling End ***********/

	/********** Comparison Start ***********/
	float MSE_rect = get_MSE(filename2, filename4);
	if (MSE_rect < MAX_MSE) {
		printf("Rectified image is equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_rect, MAX_MSE);
	}
	else {
		printf("Rectified image is NOT equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_rect, MAX_MSE);
	}
	float MSE_pool = get_MSE(filename3, filename5);
	if (MSE_pool < MAX_MSE) {
		printf("Pooled image is equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_pool, MAX_MSE);
	}
	else {
		printf("Pooled image is NOT equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_pool, MAX_MSE);
	}
	/********** Comparison End ***********/

	free(image);
	free(new_image);
	free(new_image_pool);
	return 0;
}