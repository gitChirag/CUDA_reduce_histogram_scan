#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//Create a histogram kernel
__global__ void histogram_kernel(const int* const d_container, int* d_histogram,
		const int maxsize, const int num_bins) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index > maxsize) return;
	int bin = d_container[index] % num_bins;
	atomicAdd(&d_histogram[bin], 1);
}

int main2(int argc, char **argv) {

	//Generate a random array of ints

	int maxsize = 1 << 24;
	int *container_array = new int[maxsize];
	for(int i=0;i<maxsize;i++) {
		container_array[i] = i;
	}

	const int num_bins = 10000;
	int histogram[num_bins];

	for(int i=0;i<num_bins;i++) {
		histogram[i] = 0;
	}

	int bin;
	for(int i=0;i<maxsize;i++) {
		bin = container_array[i] % num_bins;
		histogram[bin] ++;
	}


	//Allocate mem to device

	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0) {
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
				devProps.name, (int) devProps.totalGlobalMem,
				(int) devProps.major, (int) devProps.minor,
				(int) devProps.clockRate);
	}
	int* d_container_array;
	int* d_histogram;
	int* h_histogram = new int[num_bins];

	cudaMalloc(&d_container_array, sizeof(int) * maxsize);
	cudaMalloc(&d_histogram, sizeof(int) * num_bins);
	cudaMemcpy(d_container_array, container_array, sizeof(int) * maxsize, cudaMemcpyHostToDevice);
	cudaMemset(d_histogram, 0, sizeof(int) * num_bins);
	int gridsize = ceil(maxsize / 1024);
	histogram_kernel<<<gridsize, 1024>>>(d_container_array, d_histogram, maxsize, num_bins);

	cudaMemcpy(h_histogram, d_histogram, sizeof(int) * num_bins, cudaMemcpyDeviceToHost);
	//Call histogram kernel and get histogram from device to host
	//COmpare the results

	for(int i=0;i<num_bins;i++) {
		printf("%d %d\n", histogram[i], histogram[i] - h_histogram[i]);
	}

	cudaFree(d_container_array);
	cudaFree(d_histogram);
	free(h_histogram);
	free(container_array);

	return 0;
}
