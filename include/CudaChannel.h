#ifndef CUDA_CHANNEL_H
#define CUDA_CHANNEL_H

#include "Channel.h"

struct CudaChannel
{
    // Allocate a new CudaChannel on the GPU
    __host__ CudaChannel(size_t width, size_t height);

    // Copy / Move constructors
    __host__ CudaChannel(const CudaChannel& orig);
    __host__ CudaChannel(CudaChannel&& orig);

    // Copy / Move assignment operators
    __host__ CudaChannel& operator=(const CudaChannel& orig);
    __host__ CudaChannel& operator=(CudaChannel&& orig);

    // Deallocate the CudaChannel on the GPU
    __host__ ~CudaChannel();

    // Access the data (on the GPU only)
    __device__ float& operator()(const size_t x, const size_t y);
    __device__ const float& operator()(const size_t x, const size_t y) const;

    // Copy data to/from a CPU-based channel
    __host__ void copyFrom(const Channel& channel);
    __host__ void copyTo(Channel& channel) const;

    __host__ __device__ float* data();
    __host__ __device__ size_t width()  const;
    __host__ __device__ size_t height() const;

private:
    // Device variables
    float* dData;
    size_t* dWidth;
    size_t* dHeight;

    // Host variables
    size_t hWidth;
    size_t hHeight;
};


#endif
