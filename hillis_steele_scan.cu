#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void recombine_kernel(int* d_out, int* d_global_max, const int maxsize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= maxsize || blockIdx.x == 0) {
		return;
	}
	int sum = 0;
	for(int i=0;i<blockIdx.x;i++) {
		sum += d_global_max[i];
	}

	d_out[index] += sum;
}

__global__ void scan_kernel(int* d_out, int* d_global_max, const int maxsize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index >= maxsize) {
		return;
	}
	int tmp;
	for(int i = 1; i < (blockDim.x); i = i << 1) {

		tmp = 0;
		if(((index + i) < (blockIdx.x + 1) * blockDim.x) && (index + i) < maxsize) {
			tmp = d_out[index];
		}
		__syncthreads();

		if(tmp) {
			d_out[index + i] += tmp;
		}
	}
	if(threadIdx.x == blockDim.x - 1 || index == maxsize - 1) {
		d_global_max[blockIdx.x] = d_out[index];
	}
}

int main(int argc, char **argv) {

	//Generate a random array of ints

	int maxsize = 1 << 10;
	int *container_array = new int[maxsize];
	for(int i=1;i<maxsize+1;i++) {
		container_array[i-1] = i;
	}

	int* d_output;
	int *h_output = new int[maxsize];
	int *h_global_max = new int[10];
	int *d_global_max;
	int blocksize = 1024;
	int gridsize = (maxsize / blocksize) + 1;

	cudaMalloc(&d_output, sizeof(int) * maxsize);
	cudaMalloc(&d_global_max, sizeof(int) * gridsize);

	cudaMemcpy(d_output, container_array, sizeof(int) * maxsize, cudaMemcpyHostToDevice);

	scan_kernel<<<gridsize, blocksize>>>(d_output, d_global_max, maxsize);
	recombine_kernel<<<gridsize, blocksize>>>(d_output, d_global_max, maxsize);
	cudaMemcpy(h_output, d_output, sizeof(int) * maxsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_global_max, d_global_max, sizeof(int) * gridsize, cudaMemcpyDeviceToHost);
	for(int i=0;i<maxsize;i++) {
		printf("%d\n", h_output[i]);
	}

	cudaFree(d_global_max);
	cudaFree(d_output);
	free(h_output);

	return 0;

}
