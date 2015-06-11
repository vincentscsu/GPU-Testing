#include <stdio.h>
#include <cuda_runtime.h>
#include "check.h"

/* 
 * This program tests the performance of different scanning implementations,
 * such as neighbored, interleaved, and unrolled method.
 */

//Recursive reduce on host
int recursiveReduce(int *data, const int size) {
    //terminate check
    if (size == 1) return data[0];

    int stride = size / 2;

    //in-place reduce
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    //call reduce again with renewed stride
    return recursiveReduce(data, stride);
}

//Neighbored pair
__global__ void neighboredReduce(int *input, int *output, unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //set up local pointer for each block
    int *l_data = input + blockIdx.x * blockDim.x;

    //boundary check
    if (idx > n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        //convert tid to reduce divergence
        int l_idx = tid * 2 * stride;

        if (l_idx < blockDim.x) {
            l_data[l_idx] += l_data[l_idx + stride];
        }

        __syncthreads();
    }

    //write each block's final value to global memory
    if (tid == 0) {
        output[blockIdx.x] = l_data[0];
    }
}

//Interleaved pair
__global__ void interleavedReduce(int *input, int *output, unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *l_data = input + blockIdx.x * blockDim.x;

    //boundary check
    if (idx > n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            l_data[tid] += l_data[tid + stride];
        }

        __syncthreads();
    }

    //write each bllock's final value to global memroy
    if (tid == 0) {
        output[blockIdx.x] = l_data[0];
    }
}

//Completely unrolled reduce
__global__ void unrolledReduce(int *input, int *output, unsigned int n) {

    //unroll factor = 8
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *l_data = input + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int temp1 = input[idx];
        int temp2 = input[idx + blockDim.x];
        int temp3 = input[idx + blockDim.x * 2];
        int temp4 = input[idx + blockDim.x * 3];
        int temp5 = input[idx + blockDim.x * 4];
        int temp6 = input[idx + blockDim.x * 5];
        int temp7 = input[idx + blockDim.x * 6];
        int temp8 = input[idx + blockDim.x * 7];

        input[idx] = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8;
}
    __syncthreads();

    //max threads per block in both fermi or keplar is 1024
    if (blockDim.x >= 1024 && tid < 512)
        l_data[tid] += l_data[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        l_data[tid] += l_data[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        l_data[tid] += l_data[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        l_data[tid] += l_data[tid + 64];
    __syncthreads();

    //unroll each warp
    if (tid < 32) {
        //volatile forces a read/write to memory directly and not simply to cache or registers so that other threads can access at any time
        volatile int *vmem = l_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        output[blockIdx.x] = l_data[0];
}

int main(int argc, char **argv) {

    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //intialize data size
    int size = 1 << 24;
    printf("Data size: %d bytes\n", size);

    //GPU
    int blocksize = 1024;
    int gpu_sum = 0;

    //user specified block size
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size - 1) / block.x + 1, 1);

    //allocation
    size_t bytes = size * sizeof(int);
    int *h_input = (int *)malloc(bytes);
    int *h_output = (int *)malloc(grid.x * sizeof(int));
    int *h_temp = (int *)malloc(bytes);

    //initialize data
    for (int i = 0; i < size; i++) {
        h_input[i] = (int)(rand() & 0xFF);
    }

    double start, elapsed;

    int *d_input;
    int *d_output;
    CHECK(cudaMalloc((void **)&d_input, bytes));
    CHECK(cudaMalloc((void **)&d_output, grid.x * sizeof(int)));

    //host reduce
    memcpy(h_temp, h_input, bytes);
    start = seconds();
    int cpu_sum = recursiveReduce(h_temp, size);
    elapsed = seconds() - start;
    printf("CPU reduce: %d after %f sec\n", cpu_sum, elapsed);

    //neighbored reduce
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    start = seconds();
    neighboredReduce<<<grid, block>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    CHECK(cudaMemcpy(h_output, d_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_output[i];
    }

    printf("GPU neighbored reduce: %d after %f sec\n", gpu_sum, elapsed);

    //interleaved reduce
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    start = seconds();
    interleavedReduce<<<grid, block>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    CHECK(cudaMemcpy(h_output, d_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_output[i];
    }

    printf("GPU interleaved reduce: %d after %f sec\n", gpu_sum, elapsed);

    //unrolled reduce
    CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    start = seconds();
    unrolledReduce<<<grid.x / 8, block>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    CHECK(cudaMemcpy(h_output, d_output, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) {
        gpu_sum += h_output[i];
    }

    printf("GPU unrolled reduce: %d after %f sec\n", gpu_sum, elapsed);

    free(h_input);
    free(h_output);
    free(h_temp);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    CHECK(cudaDeviceReset());

    return 0;
}
