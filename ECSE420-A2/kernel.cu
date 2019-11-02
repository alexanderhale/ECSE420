#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "wm.h"
#include "A_32.h"
#include "A_512.h"
#include "A_1024.h"
#include "b_32.h"
#include "b_512.h"
#include "b_1024.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_MSE 0.00001f

__global__ void convolution(unsigned char* image, unsigned char* new_image, unsigned int matrix_dim) {
	// TODO:
		// find index of this block & thread in original image
		// find index of this block & thread in new image
	//unsigned int index = threadIdx.x + (blockIdx.x % blocks_per_row) * blockDim.x;
	//unsigned int new_index = (threadIdx.x + blockIdx.x * blockDim.x) + 4;
	
	// basic thread block indexing
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int new_index = threadIdx.x + blockIdx.x * blockDim.x;

	if (matrix_dim == 3) {
		// TODO compute convolution from old image with 3x3 convolution image (w3 from wm.h)
		for(int ii = 0; ii < 3; ii++) {
			for(int jj = 0; jj < 3; jj++) {
				new_image[new_index] += image[index] * w3[ii][jj];
			}
		}
	} else if (matrix_dim == 5) {
		// TODO 
			// compute convolution from old image with 5x5 convolution image (w5 from wm.h)
	} else {
		// TODO 
			// compute convolution from old image with 7x7 convolution image (w7 from wm.h)
	}

	// TODO
		// place result in new image
}

__global__ void mat_invert(unsigned char* A, unsigned char* x, unsigned char* b) {
	// TODO compute inverse of matrix at the index belonging to this block/thread
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
	// define number of threads
	unsigned int thread_number = 1000;		// number of threads per block we're using
	unsigned int thread_max = 1024;			// hardware limit: maximum number of threads per block
	if (thread_number > thread_max) {		// can't have more threads than the hardware limit
		thread_number = thread_max;
	}

	/***** CONVOLUTION START *****/
	// file definitions
	char* filename1 = "test.png";							// input image
	char* filename2 = "test_convolution_result.png";		// output for convolution
	char* filename3 = "test_convolve_3x3.png";				// filename for convolution comparison

	// define size of convolution matrix
	unsigned int conv_size = 3;

	// load input image
	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, filename1);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	// input image size: 
		// height x width number of pixels, 4 layers (RGBA) for each pixel, 1 char for each value
	unsigned int size_image = width * height * 4 * sizeof(unsigned char); 

	// allocate memory space on GPU
	unsigned char* cuda_image, * cuda_new_image;
	cudaMalloc((void**)& cuda_image, size_image);
	cudaMalloc((void**)& cuda_new_image, size_image);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_image, image, size_image, cudaMemcpyHostToDevice);

	// figure out how many blocks we need for this task
	unsigned int num_blocks = (size_image + thread_number - 1) / thread_number;
	// TODO update for convolution

	// call method on GPU
	convolution <<< num_blocks, thread_number >>> (cuda_image, cuda_new_image, conv_size);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned char* new_image = (unsigned char*)malloc(size_image);
	cudaMemcpy(new_image, cuda_new_image, size_image, cudaMemcpyDeviceToHost);
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

	lodepng_encode32_file(filename2, new_image, width, height);

	// comparison to check whether convolution was performed properly
	float MSE_conv = get_MSE(filename2, filename3);
	if (MSE_conv < MAX_MSE) {
		printf("Convoluted image is equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_conv, MAX_MSE);
	}
	else {
		printf("Convoluted image is NOT equal to example (MSE = %f, MAX_MSE = %f)\n", MSE_conv, MAX_MSE);
	}
	free(image);
	free(new_image);
	/********** CONVOLUTION END ***********/

	
	
	/***** MATRIX INVERSION START *****/
	// variable definition
	unsigned int* A = A_32  // or A_512 or A_1024
	unsigned int* b = b_32	// or b_512 or b_1024
	
	// allocate memory space on GPU
	unsigned int* cuda_A, * cuda_b, * cuda_x;
	cudaMalloc((void**)& cuda_A, sizeof(A));
	cudaMalloc((void**)& cuda_b, sizeof(b));
	cudaMalloc((void**)& cuda_x, sizeof(b));

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_A, A, sizeof(A), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(b), cudaMemcpyHostToDevice);

	// figure out how many blocks we need for this task
	unsigned int num_blocks = (size_image + thread_number - 1) / thread_number;
	// TODO update for matrix inversion

	// call method on GPU
	mat_invert <<< num_blocks, thread_number >>> (cuda_A, cuda_b, cuda_x);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned int* x = (unsigned int*)malloc(sizeof(b));
	cudaMemcpy(x, cuda_x, cudaMemcpyDeviceToHost);
	cudaFree(cuda_A);
	cudaFree(cuda_b);
	cudaFree(cuda_x);

	// verify that the result is correct
	// TODO verify that A*x - b = 0 (ideally in parallel)
	return 0;
}