#ifndef HOUGH_TRANSFORM_H
#define HOUGH_TRANSFORM_H

#include <vector>
#include "Channel.h"

// Represents a segment of the input image that should be processed in pixels.
struct Window
{
    size_t xMin, xMax;
    size_t yMin, yMax;
};

class HoughTransform
{
public:

    // Create a new HoughTransform. [minThetaDegrees, maxThetaDegrees] are used
    // to adjust the shape of the resulting image. 'outWidth' is the size of the
    // output image in pixels.
    HoughTransform(
        const float minThetaDegrees,
        const float maxThetaDegrees,
        const size_t outWidth
    );

    // Applies the Hough Transform to the given input channel and writes the
    // results to the provided output channel. "Window" dictates which part of
    // the input image should be examined. "minContrast" is used to determine
    // which pixels are brighter than others.
    //
    // NOTE: 'window' is deliberately copied so this function can easily be
    // called from a multi-threaded environment.
    void transform(const Channel& in, const Window window,
        const float minContrast, Channel& out);

private:

    // Helper for 'transform'. Returns true when (centerX, centerY) is brighter
    // than its neighbors. "Brighter" is measured by comparing to 'minContrast'.
    bool highContrast(const Channel& in, const int centerX,
        const int centerY, const float minContrast);

private:
    std::vector<float> mSinTable;
    std::vector<float> mCosTable;
};

#endif
