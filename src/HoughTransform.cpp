#include "HoughTransform.h"
#include <cmath>

// Make sure PI exists
#ifndef M_PI
    #define M_PI 3.14159265358979323
#endif

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
}

void HoughTransform::transform(const Channel& in, const Window window,
    const float minContrast, Channel& out)
{
    const size_t maxRadius     = (int)std::ceil(std::sqrt(in.width * in.width + in.height * in.height));
    const size_t halfRAxisSize = out.height / 2;

    for (size_t y = window.yMin; y <= window.yMax; ++y)
    {
        for (size_t x = window.xMin; x <= window.xMax; ++x)
        {
            // Only process pixels that are brighter than its surroundings
            if (highContrast(in, x, y, minContrast))
            {
                for (size_t theta = 0; theta < out.width; ++theta)
                {
                    // Calculate r using the polar normal form:
                    // (r = x*cos(theta) + y*sin(theta))
                    float r = x * mCosTable[theta] + y * mSinTable[theta];

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
                    out(theta, rLow) += weightLow;
                    out(theta, rHi)  += weightHigh;
                }
            }
        }
    }
}

bool HoughTransform::highContrast(const Channel& in, const int centerX,
    const int centerY, const float minContrast)
{
    const int width  = in.width;
    const int height = in.height;

    float centerValue = in(centerX, centerY);
    for (int y = -1; y <= 1; ++y)
    {
        int newY = centerY + y;
        if (newY < 0 || newY >= height)
            continue;

        for (int x = -1; x <= 1; ++x)
        {
            int newX = centerX + x;
            if (newX < 0 || newX >= width)
                continue;

            if (std::abs(in(newX, newY) - centerValue) >= minContrast)
                return true;
        }
    }

    return false;
}
