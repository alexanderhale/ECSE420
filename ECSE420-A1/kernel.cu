
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_MSE 0.00001f

__global__ void process_rectification(unsigned char* image, unsigned char* new_image, unsigned int size, unsigned int thread_number) {
	// process image
	//unsigned int thread_pos = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	unsigned int thread_id = (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * blockDim.x);

	for (unsigned int i = (size * thread_id)/ thread_number; i < (size *(thread_id + 1))/thread_number; i++) {
		if ((i-3)%4 !=0) {
			if (image[i] < 127) {
				new_image[i] = 127;
			}
			else {
				new_image[i] = image[i];
			}
		} else {
			new_image[i] = image[i];
		}
	}
}

__global__ void compression(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, unsigned int size, unsigned int thread_number) {

	//unsigned int thread_id = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
	unsigned int thread_id = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
	// process image
	//printf("height: %d, width: %d\n", height, width);
	for (unsigned int i = thread_id; i < size / 16; i = i + thread_number) {
		unsigned int y = (i / (width / 2)) * 2;
		unsigned int x = (i % (width / 2)) * 2;
		//printf("thread id: %d, x: %d, y: %d \n", thread_id, x, y);
		for (unsigned int type = 0; type < 4; type++) {
			unsigned int value = image[4 * width * y + 4 * x + type];
			if (value < image[4 * width * y + 4 * (x + 1) + type]) {
				value = image[4 * width * y + 4 * (x + 1) + type];
			}
			if (value < image[4 * width * (y + 1) + 4 * x + type]) {
				value = image[4 * width * (y + 1) + 4 * x + type];
			}
			if (value < image[4 * width * (y + 1) + 4 * (x + 1) + type]) {
				value = image[4 * width * (y + 1) + 4 * (x + 1) + type];
			}
			new_image[width * y + x*2 + type] = value;

			//printf("new value: %d, at coord: %d\n", value, width * y + x*2 + type);
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
	// variable definitions
	unsigned error;
	unsigned char *image, *new_image;
	unsigned char* cuda_image, * cuda_new_image;
	unsigned width, height;
	unsigned int size_image;

	// input definitions (hardcoding isn't as elegant, but is easier than passing in command-line arguments)
		// TODO add command-line arguments back in before project submission
	char* filename1 = "test.png";						// change these depending on what we're doing
	char* filename2 = "test_pooling_result.png";		// output for rectify & pooling, file2 for comparison
	char* mode = "pooling";							

	// load input image
	error = lodepng_decode32_file(&image, &width, &height, filename1);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	// define number of threads (TODO huh?)
	unsigned int thread_number = 1500;
	unsigned int thread_max = 1024;
	if (thread_number < thread_max) {
		thread_max = thread_number;
	}

	if (strcmp(mode, "rectify") == 0) {
		size_image = width * height * 4 * sizeof(unsigned char);
		new_image = (unsigned char*)malloc(size_image);

		cudaMalloc((void**)& cuda_image, size_image);
		cudaMalloc((void**)& cuda_new_image, size_image);

		cudaMemcpy(cuda_image, image, size_image, cudaMemcpyHostToDevice);

		if (thread_number > size_image) {
			thread_number = size_image;
		}
		//(unsigned int)((int))+1
		process_rectification <<<thread_number / 1024 + 1, thread_max >>>(cuda_image, cuda_new_image, size_image, thread_number);
		cudaDeviceSynchronize();
		cudaMemcpy(new_image, cuda_new_image, size_image, cudaMemcpyDeviceToHost);
		cudaFree(cuda_image);
		cudaFree(cuda_new_image);

		lodepng_encode32_file(filename2, new_image, width, height);
	}
	else if (strcmp(mode, "pooling") == 0) {
		size_image = width * height * 4 * sizeof(unsigned char);
		new_image = (unsigned char*)malloc(size_image);

		cudaMalloc((void**)& cuda_image, size_image);
		cudaMalloc((void**)& cuda_new_image, size_image);

		cudaMemcpy(cuda_image, image, size_image, cudaMemcpyHostToDevice);

		if (thread_number > size_image/16) {
			thread_number = size_image;
		}
		compression<<<thread_number / 1024 + 1, thread_max >>>(cuda_image, cuda_new_image, height, width, size_image, thread_number);
		cudaDeviceSynchronize();
		cudaMemcpy(new_image, cuda_new_image, size_image, cudaMemcpyDeviceToHost);
		cudaFree(cuda_image);
		cudaFree(cuda_new_image);

		lodepng_encode32_file(filename2, new_image, width/2, height/2);
	}

	else if (strcmp(mode, "compare") == 0) {
		// get mean squared error between image1 and image2
		float MSE = get_MSE(filename1, filename2);

		if (MSE < MAX_MSE) {
			printf("Images are equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
		}
		else {
			printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
		}
		return 0;
	}


	free(image);
	free(new_image);
	return 0;
}