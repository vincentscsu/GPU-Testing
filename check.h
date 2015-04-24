/* This is a common header file for GPU applications.
 * The purpose of this file is to make writing new programs
 * easier and faster.
 */

#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                                     \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess)                                                           \
    {                                                                                   \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                          \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
    }                                                                                   \
}

//to time the kernel
inline double seconds()
{
    struct timeval time;
    struct timezone tzone;
    gettimeofday(&time, &tzone);
    return ((double)time.tv_sec + (double)time.tv_usec*1.e-6);
}

//to determine the most computationally capable device
inline int best_device()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 1)
    {
        int maxSM = 0, best_GPU = 0;
        for (int dev = 0; dev < deviceCount; dev++)
        {
            cudaDeviceProp devProps;
            cudaGetDeviceProperties(&devProps,dev);
            if (devProps.multiProcessorCount > maxSM)
            {
                maxSM = devProps.multiProcessorCount;
                best_GPU = dev;
            }
        }

        return best_GPU;
    }
    
    else
    {
        return 0;
    }
}

//end common def
#endif
