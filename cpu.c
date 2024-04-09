
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to apply the Sobel operator to a single pixel
int sobel_pixel(int x[3][3], int y[3][3])
{
    int px = 0, py = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            px += x[i][j] * y[i][j];
            py += x[i][j] * y[j][i];
        }
    }
    return (int)sqrt(px * px + py * py);
}

// Function to apply the Sobel filter to an entire image
void sobel_filter(unsigned char *src, unsigned char *dst, int width, int height)
{
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int pixel_x[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

            // Get neighboring pixels
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if ((y + dy >= 0) && (y + dy < height) && (x + dx >= 0) && (x + dx < width))
                    {
                        pixel_x[dy + 1][dx + 1] = src[(y + dy) * width + (x + dx)];
                    }
                }
            }

            // Apply the Sobel operator to the pixel
            int val = sobel_pixel(pixel_x, Gx);
            dst[y * width + x] = (val > 255) ? 255 : (unsigned char)val;
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("7.png", &width, &height, &channels, 1); // Load image and convert to grayscale
    clock_t start, end;
    double cpu_time_used;

    if (img == NULL)
    {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    unsigned char *dst = malloc(width * height);
    if (dst == NULL)
    {
        fprintf(stderr, "Memory allocation failed for output image\n");
        stbi_image_free(img);
        exit(1);
    }
    start = clock();
    sobel_filter(img, dst, width, height);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Sobel filter took %f seconds to execute \n", cpu_time_used);
    stbi_write_png("output.png", width, height, 1, dst, width);

    stbi_image_free(img);
    free(dst);

    return 0;
}
