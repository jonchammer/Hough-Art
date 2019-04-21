#include "Channel.h"

Channel::Channel(size_t width, size_t height) :
    data(width * height),
    width(width),
    height(height)
{}

void Channel::fill(float value)
{
    for (size_t i = 0; i < width * height; ++i)
        data[i] = value;
}

float& Channel::operator()(const size_t x, const size_t y)
{
    return data[y * width + x];
}

const float& Channel::operator()(const size_t x, const size_t y) const
{
    return data[y * width + x];
}
