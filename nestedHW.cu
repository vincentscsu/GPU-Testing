#include <stdio.h>
#include <cuda_runtime.h>
#include "check.h"

/*
 * This program tests dinamic parallelism by implementing nested hello world.
 * Must be linked with device runtime library at compile time(-l), and specify
 * architecture (arch). Kernel launch from __device__ and __global__ requires separate
 * compilation modes (rdc):
 * nvcc -arch=sm_35 -rdc=true nestedHW.cu -o nestedHW -lcudadevrt
 */

__global__ void nestedHW(int const threads, int depth) {

    int tid = threadIdx.x;

    printf("Hey you! from layer %d block %d thread %d\n", depth, blockIdx.x, tid);

    //terminate when the last child grid spawns only one thread
    if (threads == 1)
        return;

    int new_threads = threads >> 1;

    if (tid == 0 && new_threads != 0) {
        //depth doesn't really do anything, only for visualization
        depth += 1;
        nestedHW<<<1, new_threads>>>(new_threads, depth);
        printf("----------Depth %d----------\n", depth);
    }
}

int main(int argc, char **argv) {

    //choose a GPU whose compute capability is > 3.5 (required for dynamic parallelism)
    int deviceCount = 0;

    //check how many cuda devices found
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("There is no cuda-capable device on this system.\n");
        return 0;
    }

    //flag to check if such GPU exists in the system
    int flag = 0;

    for (int i = 0; i < deviceCount; i++) {

        int dev = i;

        CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));

        //capability
        if ((deviceProp.major == 3 && deviceProp.minor >= 5) || (deviceProp.major > 3)) {
            //device name
            printf("Device %d: %s \n", dev, deviceProp.name);
            //capability
            printf("CUDA Capability Major/Minor %d.%d\n", deviceProp.major, deviceProp.minor);

            CHECK(cudaSetDevice(dev));
            flag = 1;
            continue;
        }
    }

    if (flag == 0) {
        printf("There is no GPU that can handle dynamic parallelism in this system :(\n");
        return 0;
    }

    //total threads in the grid
    int total_threads = 8;
    //number of threads per block
    int block_size = 8;
    //number of blocks per grid
    int grid_size = 1;

    //user specified block numbers
    if (argc > 1) {
        grid_size = atoi(argv[1]);
        total_threads = grid_size * block_size;
    }

    dim3 block (block_size, 1);
    dim3 grid ((total_threads - 1) / block_size + 1, 1);

    printf("Nested Hello World with grid %d block %d\n", grid.x, block.x);

    //pass in depth 0 as the first execution
    nestedHW<<<grid, block>>>(block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());

    return 0;
}
