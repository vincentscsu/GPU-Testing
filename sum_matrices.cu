#include <stdio.h>
#include <cuda_runtime.h>
#include "check.h"

/* 
 * This program computes the sum of two square matrices each with 16M elements
 * to test the GPU's compute capability. It currently includes 1D grid 1D block, 2D grid 1D block,
 * and 2D grid 2D block methods. User can specify the number of threads per block in the following
 * oreder: ./<prog> <block.x for 1D kernel> <block.x for 2D 1D kernel> <block.x for 2D 2D kernel>
 * <block.y for 2D 2D kernel>.
 */

//Host summation function
void sum_CPU(float *A, float *B, float *C, int x, int y)
{
    int i, j;

    float *a = A;
    float *b = B;
    float *c = C;

    //sum on each row
    for (j = 0; j < y; j++)
    {
        for (i = 0; i < x; i++)
        {
            c[i] = a[i] + b[i];
        }

        //go to next row
        a += x;
        b += x;
        c += x;
    }
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

//GPU summation kernel - 1D
__global__ void sum_GPU_1D(float *A, float *B, float *C, int x, int y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = 0;

    if (i < x)
    {
        for (j = 0; j < x; j++)
        {
            int idx = j*x + i;
            C[idx] = A[idx] + B[idx];
        }
    }
}

//GPU summation kernel - 2D grid 1D block
__global__ void sum_GPU_2D_1D(float *A, float *B, float *C, int x, int y)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*x + ix;

    if (ix < x && iy < y)
    {
        C[idx] = A[idx] + B[idx];
    }
}

//GPU summation kernel - 2D grid 2D block
__global__ void sum_GPU_2D_2D(float *A, float *B, float *C, int x, int y)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    int idx = iy*x + ix;

    if (ix < x && iy < y)
    {
        C[idx] = A[idx] + B[idx];
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
        printf("Results from CPU and GPU do not match at %d (%f, %f)\n", i, A[i], B[i]);
    }
}

int main(int argc, char *argv[])
{
    //return device info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //initialize data size;
    int nx = 1 << 14;
    int ny = 1 << 14;
    printf("Matrix size: %d, %d\n", nx, ny);
    
    int nelem = nx*ny;

    int size = nelem*sizeof(float);

    //allocate host memory
    float *h_A, *h_B, *h_C, *cpu_result;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    cpu_result = (float*)malloc(size);

    //initialize data on host
    data_init(h_A, nelem);
    data_init(h_B, nelem);
    memset(h_C, 0, size);
    memset(cpu_result, 0 ,size);
    
    //matrix summation on host
    double start = seconds();
    sum_CPU(h_A, h_B, cpu_result, nx, ny);
    double elapsed = seconds() - start;
    printf("sum_CPU used: %f sec\n", elapsed);

    //allocate device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, size));
    CHECK(cudaMalloc((float**)&d_B, size));
    CHECK(cudaMalloc((float**)&d_C, size));

    //copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    //kernel specs for 1D
    int block_dim_x = 256;
    if (argc > 1)
    {
        block_dim_x = atoi(argv[1]);
    }
    dim3 block (block_dim_x,1,1);
    dim3 grid (((nx-1)/block.x)+1, 1 ,1);
    
    //get ready to time the kernel
    start = seconds();

    //launch kernel
    sum_GPU_1D<<<grid,block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    printf("sum_GPU_1D<<< %d , %d >>> used: %f sec\n", grid.x, block.x, elapsed);

    //check kernel errors
    CHECK(cudaGetLastError());

    //copy result from device to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    //check results
    check_result(cpu_result, h_C, nelem);

    //kernel specs for 2D 1D
    block_dim_x = 32;
    if (argc > 2)
    {
        block_dim_x = atoi(argv[2]);
    }
    dim3 block_2(block_dim_x, 1, 1);
    dim3 grid_2((nx-1)/block.x+1,ny,1);

    //get ready to time the kernel
    start = seconds();

    //launch kernel
    sum_GPU_2D_1D<<<grid_2,block_2>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    printf("sum_GPU_2D_1D<<< (%d , %d), (%d , %d) >>> used: %f sec\n", grid_2.x, grid_2.y, block_2.x, block_2.y, elapsed);

    //check kernel errors
    CHECK(cudaGetLastError());

    //copy result from device to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    //check results
    check_result(cpu_result, h_C, nelem);

    //kernel specs for 2D 2D
    int block_dim_y;
    block_dim_x = 32;
    block_dim_y = 32;
    if (argc > 4)
    {
        block_dim_x = atoi(argv[3]);
        block_dim_y = atoi(argv[4]);
    }
    dim3 block_3(block_dim_x, block_dim_y, 1);
    dim3 grid_3((nx-1)/block.x+1, (ny-1)/block.y+1, 1);

    //get ready to time the kernel
    start = seconds();

    //launch kernel
    sum_GPU_2D_2D<<<grid_3,block_3>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    elapsed = seconds() - start;
    printf("sum_GPU_2D_2D<<< (%d , %d), (%d , %d) >>> used: %f sec\n", grid_3.x, grid_3.y, block_3.x, block_3.y, elapsed);

    //check kernel errors
    CHECK(cudaGetLastError());

    //copy result from device to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    //check results
    check_result(cpu_result, h_C, nelem);
    //free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    //free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(cpu_result);

    //reset device
    CHECK(cudaDeviceReset());

    return 0;
}





