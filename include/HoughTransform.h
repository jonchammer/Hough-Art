#ifndef HOUGH_TRANSFORM_H
#define HOUGH_TRANSFORM_H

#include <cmath>

// Make sure PI exists
#ifndef M_PI
    #define M_PI 3.14159265358979323
#endif

// Represents a segment of the input image that should be processed in pixels.
struct Window
{
    size_t xMin, xMax;
    size_t yMin, yMax;
};

template <class T>
class HoughTransform
{
public:

    // Create a new HoughTransform. [minThetaDegrees, maxThetaDegrees] are used
    // to adjust the shape of the resulting image. 'outWidth' is the size of the
    // output image in pixels.
    HoughTransform(const T minThetaDegrees, const T maxThetaDegrees, const size_t outWidth)
    {
        // X output ranges from minTheta to maxTheta degrees
        // Y output ranges from -maxRadius to maxRadius
        const T minThetaRadians = M_PI / T{180.0} * minThetaDegrees;
        const T maxThetaRadians = M_PI / T{180.0} * maxThetaDegrees;
        const T range           = maxThetaRadians - minThetaRadians;

        // Precompute the sin and cos values we will use for each value of theta
        mSinTable = new T[outWidth];
        mCosTable = new T[outWidth];
        for (size_t theta = 0; theta < outWidth; ++theta)
        {
            // Map theta from [0, width] -> [minThetaRadians, maxThetaRadians]
            T thetaRadians   = (theta * range) / outWidth + minThetaRadians;
            mSinTable[theta] = sin(thetaRadians);
            mCosTable[theta] = cos(thetaRadians);
        }
    }

    ~HoughTransform()
    {
        // Clean up the precomputed sin/cos tables
        delete[] mSinTable;
        delete[] mCosTable;
        mSinTable = nullptr;
        mCosTable = nullptr;
    }

    // Applies the Hough Transform to the given input channel and writes the
    // results to the provided output channel. "Window" dictates which part of
    // the input image should be examined. "minContrast" is used to determine
    // which pixels are brighter than others.
    //
    // NOTE: 'window' is deliberately copied so this function can easily be
    // called from a multi-threaded environment.
    void transform(const Channel<T>& in, const Window window,
        const T minContrast, Channel<T>& out)
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
                        T r = mCosTable[theta] * x + mSinTable[theta] * y;

                        // Scale r to the range [-maxRadius/2, maxRadius/2]
                        // To avoid discretization artifacts, we add a bit to both
                        // cells covered by val. The weight is proportional to how
                        // close we are to the lower bound or the upper bound.
                        T val    = (r * halfRAxisSize / maxRadius) + halfRAxisSize;
                        int rLow = (int) std::floor(val);
                        int rHi  = (int) std::ceil(val);

                        T weightHigh = val - (int) val;
                        T weightLow  = T{1.0} - weightHigh;

                        // Update the corresponding slot in the accumulator buffer
                        out(theta, rLow) += weightLow;
                        out(theta, rHi)  += weightHigh;
                    }
                }
            }
        }
    }

private:

    // Helper for 'transform'. Returns true when (centerX, centerY) is brighter
    // than its neighbors. "Brighter" is measured by comparing to 'minContrast'.
    bool highContrast(const Channel<T>& in, const int centerX,
        const int centerY, const T minContrast)
    {
        const int width  = in.width;
        const int height = in.height;

        T centerValue = in(centerX, centerY);
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

private:
    T* mSinTable;
    T* mCosTable;
};

#endif
