#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <cstring>
#include "il.h"

using std::string;

// Useful typedef
typedef unsigned char byte;

// Represents a single channel of an image.
template <class T>
struct Channel
{
    T* data;
    size_t width;
    size_t height;
    size_t stride;

    T& operator()(const size_t x, const size_t y)
    {
        return data[(y * width + x) * stride];
    }

    const T& operator()(const size_t x, const size_t y) const
    {
        return data[(y * width + x) * stride];
    }
};

// This class is a C++ wrapper around the image functionality provided by
// DevIL. Make sure to call Image::init() before this class is used.
class Image
{
public:
    // Constructors / Destructors
    Image();
    Image(size_t width, size_t height);
    Image(const Image& other);
    ~Image();

    // File I/O
    bool load(const string& filename);
    bool save(const string& filename);

    // Initialize DevIL - This function MUST be called before images can be
    // loaded or saved.
    static void init()
    {
        ilInit();
        ilSetInteger(IL_JPG_QUALITY, 99);
    }

    // Creates a new Channel object using information from this image. This
    // Channel will be normalized to the range [0, 1]. Channels [0, 1, 2, 3]
    // correspond to the [R, G, B, A] colors.
    template <class T>
    Channel<T> readChannel(const size_t channel) const
    {
        Channel<T> result;
        result.width  = mWidth;
        result.height = mHeight;
        result.stride = 1;
        result.data   = new T[mWidth * mHeight];

        for (size_t i = 0; i < mWidth * mHeight; ++i)
        {
            result.data[i] = T(mData[4 * i + channel]) / T{255.0};
        }

        return result;
    }

    // Writes the information from the given channel object back to the image.
    // The channel is assumed to be normalized to the range [0, 1]. Channels
    // [0, 1, 2, 3] correspond to the [R, G, B, A] colors.
    template <class T>
    void writeChannel(const Channel<T>& data, const size_t channel)
    {
        for (size_t i = 0; i < mWidth * mHeight; ++i)
        {
            mData[4 * i + channel] = (byte)(data.data[i] * 255.0);
        }
    }

    // Getters / Setters
    size_t getWidth() const;
    size_t getHeight() const;
    const byte* getData() const;
    byte* getData();

private:
    ILuint mID;
    size_t mWidth, mHeight;
    byte* mData;
    bool mOwnData;
};

#endif
