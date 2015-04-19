/* This is a common header file for all my GPU testing programs. 
 * The purpose of this file is to make writing new testing programs
 * easier.
 * It currently includes the check function of cuda calls,
 * and the time function to test the elapsed time for each kernel.
 * */

#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));\
    }\
}

inline double seconds()
{
    struct timeval time;
    struct timezone tzone;
    gettimeofday(&time, &tzone);
    return ((double)time.tv_sec + (double)time.tv_usec*1.e-6);
}

//end common def
#endif
