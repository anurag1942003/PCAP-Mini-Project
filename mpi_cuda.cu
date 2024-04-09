//%%writefile final.cu
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__device__ float clamp(float val, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(maxVal, val));
}

__global__ void sobelEdgeDetection(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float edgeX = 0.0;
    float edgeY = 0.0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                float val = (float)input[(y + ky) * width + (x + kx)];
                edgeX += Gx[ky + 1][kx + 1] * val;
                edgeY += Gy[ky + 1][kx + 1] * val;
            }
        }
    }

    float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
    edge = clamp(edge, 0, 255);
    output[y * width + x] = (unsigned char)edge;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, channels;
    unsigned char *img = NULL, *sub_img, *sub_img_result;
    int sub_height, sub_size;

    if (rank == 0) {
        img = stbi_load("1.png", &width, &height, &channels, 1); // Load image and convert to grayscale
        if (img == NULL) {
            fprintf(stderr, "Error in loading the image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image width and height
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate size of each sub-image
    sub_height = height / size;
    sub_size = width * sub_height;

    sub_img = (unsigned char *)malloc(sub_size);
    sub_img_result = (unsigned char *)malloc(sub_size);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start timing simultaneously
    double start_time = MPI_Wtime();
    // Scatter the image to all processes
    MPI_Scatter(img, sub_size, MPI_UNSIGNED_CHAR, sub_img, sub_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, sub_size);
    cudaMalloc(&d_output, sub_size);

    // Copy sub-image to device
    cudaMemcpy(d_input, sub_img, sub_size, cudaMemcpyHostToDevice);

    dim3 blocks((width + 15) / 16, (sub_height + 15) / 16);
    dim3 threadsPerBlock(16, 16);

    // Launch CUDA kernel
    sobelEdgeDetection<<<blocks, threadsPerBlock>>>(d_input, d_output, width, sub_height);
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Copy processed sub-image back to host
    cudaMemcpy(sub_img_result, d_output, sub_size, cudaMemcpyDeviceToHost);

    // Gather processed sub-images at root
    MPI_Gather(sub_img_result, sub_size, MPI_UNSIGNED_CHAR, img, sub_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish before stopping the timer
    double end_time = MPI_Wtime();    
    if (rank == 0) {
        // Only the root process saves the final image
        printf("Edge detection took %f seconds.\n", end_time - start_time);
        stbi_write_png("output.png", width, height, 1, img, width);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    if (img) {
        stbi_image_free(img);
    }
    free(sub_img);
    free(sub_img_result);

    MPI_Finalize();
    return 0;
}
