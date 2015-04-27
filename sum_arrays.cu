#include <stdio.h>
#include <cuda_runtime.h>
#include "check.h"

/*
 * This program generates two arrays of 16M elements to test the device's
 * cpability.
 */

//Host summation funciton
void sum_CPU(float *A, float *B, float *C, int n)
{
    int i;

    for (i = 0;i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

//GPU summation kernel
__global__ void sum_GPU(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    C[i] = A[i] + B[i];
}

//random data generation
void data_init(float *data, int size)
{   
    //seed for random generation
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        //mask off to get only lower two bytes
        data[i] = (float)(rand() & 0xFF);
    }
}

//check results
void check_result(float *A, float *B, int size)
{
    double eps = 1.0e-8;
    int i;
    int flag = 1;

    for (i = 0; i < size; i++)
    {
        if (abs(A[i] - B[i]) > eps)
        {   
            flag = 0;
            break;
        }
    }
    if (flag == 1)
    {
        printf("Results match!\n");
    }
    else
    {
        printf("Results from CPU and GPU do not match at %d (%f, %f)\n",
                i, A[i], B[i]);
    }
}


int main()
{
    //return device info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //initialize data size;
    int size = 1 << 24;
    printf("Array size: %d elements\n", size);
    
    //allocate host memory
    float *h_A, *h_B, *h_C, *cpu_result;
    h_A = (float*)malloc(size*sizeof(float));
    h_B = (float*)malloc(size*sizeof(float));
    h_C = (float*)malloc(size*sizeof(float));
    cpu_result = (float*)malloc(size*sizeof(float));

    //initialize data on host
    data_init(h_A, size);
    data_init(h_B, size);
    memset(h_C, 0, size);
    memset(cpu_result, 0 ,size);

    //get ready to time cpu summation
    double start, elapsed;

    //summation on host for reference
    start = seconds();
    sum_CPU(h_A, h_B, cpu_result, size);
    elapsed = seconds() - start;
    printf("sum_CPU uesed: %f sec\n", elapsed);
    
    //allocate device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, size*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_B, size*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_C, size*sizeof(float)));

    //copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice));

    //kernel specs
    dim3 block (1024,1,1);
    dim3 grid (((size-1)/block.x)+1, 1 ,1);
    
    //get ready to time the kernel
    start = seconds();

    //launch kernel
    sum_GPU<<<grid,block>>>(d_A, d_B, d_C, size);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    printf("sum_GPU<<< %d , %d >>> used: %f sec\n", grid.x, block.x, elapsed);

    //check kernel errors
    CHECK(cudaGetLastError());

    //copy result from device to host
    CHECK(cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost));

    //check results
    check_result(cpu_result, h_C, size);

    //free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    //free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(cpu_result);

    return 0;
}

