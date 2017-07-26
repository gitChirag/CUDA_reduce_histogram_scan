#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
__global__ void shmem_reduce_kernel(double * d_out, const double * d_in) {
	extern __shared__ double sdata[];
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[myId];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = sdata[0];
	}
}

void reduce(double * d_out, double * d_intermediate, double * d_in, int size,
		bool usesSharedMemory) {
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;
	int tmp;
	while (1) {
		shmem_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(
					d_intermediate, d_in);
		cudaDeviceSynchronize();
		tmp = blocks;
		blocks = ceil (blocks / threads);
		if (blocks == 1 && tmp % threads == 0) {
			break;
		}
		if (blocks == 0) {
			threads = tmp % threads;
			break;
		}
		cudaMemcpy(d_in, d_intermediate, size * sizeof(double),
				cudaMemcpyDeviceToDevice);
	}

	blocks = 1;
	shmem_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_out,
				d_intermediate);
}

int main3(int argc, char **argv) {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
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

	const int ARRAY_SIZE = 1<<25;

	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(double);
	// generate the input array on the host
	double *h_in = new double[ARRAY_SIZE];
	double sum = 0.0f;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = i;
	}

	double start_time = clock();

	for (int i = 0; i < ARRAY_SIZE; i++) {
		sum = h_in[i] > sum ? h_in[i] : sum;
	}

	double end_time = clock();
	printf("CPU Time: %lf\n", (end_time - start_time) / CLOCKS_PER_SEC * 1000);
	// declare GPU memory pointers
	double * d_in, *d_intermediate, *d_out;

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
	cudaMalloc((void **) &d_out, sizeof(double));

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// launch the kernel

	printf("Running reduce with shared mem\n");
	cudaEventRecord(start, 0);
	for (int i = 0; i < 1; i++) {
		reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
	}
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	// copy back the sum from GPU
	double h_out;
	cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
	printf("average time elapsed: %f\n", elapsedTime);
	printf("Sum: %f %f", h_out, sum);
	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_intermediate);
	cudaFree(d_out);
	return 0;
}
