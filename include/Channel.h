#ifndef CHANNEL_H
#define CHANNEL_H

#include <cstddef>
#include <vector>

// Represents a single channel of an image.
struct Channel
{
    std::vector<float> data;
    size_t width;
    size_t height;

    Channel(size_t width, size_t height);

    void fill(float value);

    float& operator()(const size_t x, const size_t y);
    const float& operator()(const size_t x, const size_t y) const;
};

#endif
