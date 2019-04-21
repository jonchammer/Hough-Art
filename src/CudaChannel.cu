#include "CudaChannel.h"
#include "CudaCommon.h"

__host__
CudaChannel::CudaChannel(size_t w, size_t h) :
    dData(nullptr),
    dWidth(nullptr),
    dHeight(nullptr),
    hWidth(w),
    hHeight(h)
{
    gpuErrchk( cudaMalloc(&dData,   w * h * sizeof(float)) );
    gpuErrchk( cudaMalloc(&dWidth,  sizeof(size_t)) );
    gpuErrchk( cudaMalloc(&dHeight, sizeof(size_t)) );

    gpuErrchk( cudaMemcpy(dWidth,  &hWidth,  sizeof(size_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dHeight, &hHeight, sizeof(size_t), cudaMemcpyHostToDevice) );
}

__host__
CudaChannel::CudaChannel(const CudaChannel& channel) :
    dData(nullptr),
    dWidth(nullptr),
    dHeight(nullptr),
    hWidth(channel.hWidth),
    hHeight(channel.hHeight)
{
    const size_t bytes = hWidth * hHeight * sizeof(float);

    // Allocate memory
    gpuErrchk( cudaMalloc(&dData,   bytes)          );
    gpuErrchk( cudaMalloc(&dWidth,  sizeof(size_t)) );
    gpuErrchk( cudaMalloc(&dHeight, sizeof(size_t)) );

    // Copy memory
    gpuErrchk( cudaMemcpy(dData, channel.dData, bytes, cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpy(dWidth,  &hWidth,  sizeof(size_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dHeight, &hHeight, sizeof(size_t), cudaMemcpyHostToDevice) );
}

__host__
CudaChannel::CudaChannel(CudaChannel&& channel) :
    dData(channel.dData),
    dWidth(channel.dWidth),
    dHeight(channel.dHeight),
    hWidth(channel.hWidth),
    hHeight(channel.hHeight)
{
    channel.dData   = nullptr;
    channel.dWidth  = nullptr;
    channel.dHeight = nullptr;
    channel.hWidth  = 0;
    channel.hHeight = 0;
}

__host__
CudaChannel& CudaChannel::operator=(const CudaChannel& orig)
{
    // Delete any old data
    gpuErrchk( cudaFree(dData)   ); dData   = nullptr;
    gpuErrchk( cudaFree(dWidth)  ); dWidth  = nullptr;
    gpuErrchk( cudaFree(dHeight) ); dHeight = nullptr;

    hWidth  = orig.hWidth;
    hHeight = orig.hHeight;
    const size_t bytes = hWidth * hHeight * sizeof(float);

    // Allocate new memory
    gpuErrchk( cudaMalloc(&dData,   bytes)          );
    gpuErrchk( cudaMalloc(&dWidth,  sizeof(size_t)) );
    gpuErrchk( cudaMalloc(&dHeight, sizeof(size_t)) );

    // Copy new data
    gpuErrchk( cudaMemcpy(dData,   orig.dData, bytes, cudaMemcpyDeviceToDevice)      );
    gpuErrchk( cudaMemcpy(dWidth,  &hWidth,  sizeof(size_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dHeight, &hHeight, sizeof(size_t), cudaMemcpyHostToDevice) );

    return *this;
}

__host__
CudaChannel& CudaChannel::operator=(CudaChannel&& orig)
{
    // Move original data
    dData   = orig.dData;
    dWidth  = orig.dWidth;
    dHeight = orig.dHeight;
    hWidth  = orig.hWidth;
    hHeight = orig.hHeight;

    // Set orig to a default state
    orig.dData   = nullptr;
    orig.dWidth  = nullptr;
    orig.dHeight = nullptr;
    orig.hWidth  = 0;
    orig.hHeight = 0;

    return *this;
}

__host__
CudaChannel::~CudaChannel()
{
    gpuErrchk( cudaFree(dData)   );
    gpuErrchk( cudaFree(dWidth)  );
    gpuErrchk( cudaFree(dHeight) );
}

__device__
float& CudaChannel::operator()(const size_t x, const size_t y)
{
    return dData[y * width() + x];
}

__device__
const float& CudaChannel::operator()(const size_t x, const size_t y) const
{
    return dData[y * width() + x];
}

__host__
void CudaChannel::copyFrom(const Channel& channel)
{
    gpuErrchk( cudaMemcpy(
        dData,
        channel.data.data(),
        width() * height() * sizeof(float),
        cudaMemcpyHostToDevice
    ) );
}

__host__
void CudaChannel::copyTo(Channel& channel) const
{
    gpuErrchk( cudaMemcpy(
        channel.data.data(),
        dData,
        width() * height() * sizeof(float),
        cudaMemcpyDeviceToHost
    ) );
}

__host__ __device__
float* CudaChannel::data()
{
    return dData;
}

__host__ __device__
size_t CudaChannel::width()  const
{
    size_t result;
    #ifdef __CUDA_ARCH__
        result = *dWidth;
    #else
        result = hWidth;
    #endif
    return result;
}

__host__ __device__
size_t CudaChannel::height() const
{
    size_t result;
    #ifdef __CUDA_ARCH__
        result = *dHeight;
    #else
        result = hHeight;
    #endif
    return result;
}
