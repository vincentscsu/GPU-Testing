#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "check.h"
/* 
 * This program returns important information for 
 * each cuda-capable device on this system/server
 */

int main(int argc, char **argv)
{
    int deviceCount = 0;
    
    //check how many cuda devices found
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("There is no cuda-capable device on this system.\n");
    }
    else
    {
        printf("There are %d cuda-capable device(s) on this system.\n", deviceCount);
    }

    for (int i = 0; i < deviceCount; i++)
    {
        printf("\n\n");

        int dev = i;
        int driverVersion = 0, runtimeVersion = 0;

        CHECK(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        
        //device name
        printf("Device %d: %s \n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        //runtime and driver version
        printf("CUDA Driver Version/Runtime Version       %d.%d/%d.%d\n",
                driverVersion/1000, (driverVersion%100)/10, 
                runtimeVersion/1000, (runtimeVersion%100)/10);
        //capability
        printf("CUDA Capability Major/Minor               %d.%d\n",
                deviceProp.major, deviceProp.minor);
        //SM counts
        printf("Number of Multprocessors                  %d\n",
                deviceProp.multiProcessorCount);
        //warp size
        printf("Warp Size                                 %d\n",
                deviceProp.warpSize);
        //maximum threads per SM
        printf("Maximu Number of Threads per SM           %d\n",
                deviceProp.maxThreadsPerMultiProcessor);
        //maximum threads per block
        printf("Maximu Number of Threads per block        %d\n",
                deviceProp.maxThreadsPerMultiProcessor);
        //max block size
        printf("Maximum Dimension Per Block               %d x %d x %d\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
        //max grid size
        printf("Maximum Dimension Per Grid                %d x %d x %d\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
        //global memory
        printf("Total Global Memory                       %.2f MBytes (%llu bytes)\n",
                (float) deviceProp.totalGlobalMem/pow(1024.0,3),
                (unsigned long long)deviceProp.totalGlobalMem);
        //constant memory
        printf("Total Constant Memory                     %lu bytes\n",
                (unsigned long)(deviceProp.totalConstMem));
        //shared memory
        printf("Shared Memory Per Block                   %lu bytes\n",
                (unsigned long)(deviceProp.sharedMemPerBlock));
        //registers per block
        printf("Registers Available Per Block             %d\n",
                deviceProp.regsPerBlock);
        //GPU clock rate
        printf("GPU Clock Rate                            %.0f Mhz (%0.2f GHz)\n",
                deviceProp.clockRate*1.e-3f, deviceProp.clockRate*1.e-6f);
        //Memory clock rate
        printf("Memory Clock Rate                         %.0f Mhz\n",
                deviceProp.memoryClockRate*1e-3f);
        //bandwidth
        printf("Memory Bus Width                          %d-bit\n", 
                deviceProp.memoryBusWidth);
        //L2 cache size
        if (deviceProp.l2CacheSize)
        {
            printf("L2 Cache Size                         %d bytes\n",
            deviceProp.l2CacheSize);
        }
        //maximum texture dimension
        printf("Max Texture Dimension Size (x,y,z)        1D = (%d), 2D = (%d,%d), 3D = (%d,%d,%d)\n",
                deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
                deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        //maximum layered texture size
        printf("Max Layered Textrue Size (dim) x layers   1D = (%d) x %d, 2D = (%d,%d) x %d\n",
                deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
                deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
        //max memory pitch
        printf("Max Memory Pitch                          %lu bytes\n",
                (unsigned long)(deviceProp.memPitch));
    }
    
    int best = best_device();
    printf("\nMost powerful device is device %d\n", best);

    return 0;
}
