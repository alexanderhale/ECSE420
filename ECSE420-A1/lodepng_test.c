/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

void process(char* input_filename, char* output_filename)
{
  unsigned error;
  unsigned char *image, *new_image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  new_image = malloc(width * height * 4 * sizeof(unsigned char));

  // process image
  unsigned char value;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {	

    	value = image[4*width*i + 4*j];

	    new_image[4*width*i + 4*j + 0] = value; // R
	    new_image[4*width*i + 4*j + 1] = value; // G
	    new_image[4*width*i + 4*j + 2] = value; // B
	    new_image[4*width*i + 4*j + 3] = image[4*width*i + 4*j + 3]; // A
    }
  }

  lodepng_encode32_file(output_filename, new_image, width, height);

  free(image);
  free(new_image);
}

int main(int argc, char *argv[])
{
  char* input_filename = argv[1];
  char* output_filename = argv[2];

  process(input_filename, output_filename);

  return 0;
}
