#include "HoughTransform.h"
#include "CudaChannel.h"
#include "CudaCommon.h"
#include <cmath>

// Make sure PI exists
#ifndef M_PI
    #define M_PI 3.14159265358979323
#endif

// Max size for constant memory is 64KB. Therefore, the largest images we can
// process are 64,000 / 2 tables / sizeof(float) = 8000 pixels wide
__constant__ float sinTable[8000];
__constant__ float cosTable[8000];

namespace
{
    __device__ bool highContrast(const float* in, int w, int h, int cx, int cy, float minContrast)
    {
        float centerValue = in[cy * w + cx];
        for (int y = -1; y <= 1; ++y)
        {
            int newY = cy + y;
            if (newY < 0 || newY >= h)
                continue;

            for (int x = -1; x <= 1; ++x)
            {
                int newX = cx + x;
                if (newX < 0 || newX >= w)
                    continue;

                if (std::abs(in[newY * w + newX]/*in(newX, newY)*/ - centerValue) >= minContrast)
                    return true;
            }
        }

        return false;
    }

    __global__ void houghKernel(
        float* inData,
        int inWidth,
        int inHeight,
        float minContrast,
        float halfRAxisSize,
        float maxRadius,
        float* outData,
        int outWidth,
        int outHeight
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Only process pixels that are in the image and are brighter than
        // its surroundings
        if (x < inWidth && y < inHeight && highContrast(inData, inWidth, inHeight, x, y, minContrast))
        {
            for (size_t theta = 0; theta < outWidth; ++theta)
            {
                // Calculate r using the polar normal form:
                // (r = x*cos(theta) + y*sin(theta))
                float r = x * cosTable[theta] + y * sinTable[theta];

                // Scale r to the range [-maxRadius/2, maxRadius/2]
                // To avoid discretization artifacts, we add a bit to both
                // cells covered by val. The weight is proportional to how
                // close we are to the lower bound or the upper bound.
                float val = (r * halfRAxisSize / maxRadius) + halfRAxisSize;
                int rLow  = (int) std::floor(val);
                int rHi   = (int) std::ceil(val);

                float weightHigh = val - (int) val;
                float weightLow  = 1.0f - weightHigh;

                // Update the corresponding slot in the accumulator buffer
                outData[rLow * outWidth + theta] += weightLow;
                outData[rHi * outWidth + theta]  += weightHigh;
            }
        }
    }
}

HoughTransform::HoughTransform(
    const float minThetaDegrees,
    const float maxThetaDegrees,
    const size_t outWidth
) :
    mSinTable(outWidth),
    mCosTable(outWidth)
{
    // X output ranges from minTheta to maxTheta degrees
    // Y output ranges from -maxRadius to maxRadius
    const float minThetaRadians = M_PI / 180.0f * minThetaDegrees;
    const float maxThetaRadians = M_PI / 180.0f * maxThetaDegrees;
    const float range           = maxThetaRadians - minThetaRadians;

    // Precompute the sin and cos values we will use for each value of theta
    for (size_t theta = 0; theta < outWidth; ++theta)
    {
        // Map theta from [0, width] -> [minThetaRadians, maxThetaRadians]
        float thetaRadians = (theta * range) / outWidth + minThetaRadians;
        mSinTable[theta]   = sin(thetaRadians);
        mCosTable[theta]   = cos(thetaRadians);
    }

    // Copy tables to constant memory
    gpuErrchk( cudaMemcpyToSymbol(sinTable, mSinTable.data(), sizeof(float) * mSinTable.size()) );
    gpuErrchk( cudaMemcpyToSymbol(cosTable, mCosTable.data(), sizeof(float) * mCosTable.size()) );
}

void HoughTransform::transform(const Channel& in, const Window window,
    const float minContrast, Channel& out)
{
    const size_t maxRadius     = (int)std::ceil(std::sqrt(in.width * in.width + in.height * in.height));
    const size_t halfRAxisSize = out.height / 2;

    // Copy data to Cuda
    CudaChannel cIn(in.width, in.height);
    CudaChannel cOut(out.width, out.height);
    cIn.copyFrom(in);

    // Calculate block / thread sizes
    const dim3 threadDimensions(32, 32);
    const dim3 grid(
        (in.width  + threadDimensions.x - 1) / threadDimensions.x,
        (in.height + threadDimensions.y - 1) / threadDimensions.y
    );

    // Execute kernel on GPU
    houghKernel<<<grid, threadDimensions>>>(
        cIn.data(), cIn.width(), cIn.height(),
        minContrast, halfRAxisSize, maxRadius,
        cOut.data(), cOut.width(), cOut.height()
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy data back to CPU
    cOut.copyTo(out);
}
