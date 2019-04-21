#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <cstring>
#include <cstdint>

#include <il.h>
#include "Channel.h"

using std::string;

// Useful typedef
typedef uint8_t byte;

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
    Channel readChannel(const size_t channel) const;

    // Writes the information from the given channel object back to the image.
    // The channel is assumed to be normalized to the range [0, 1]. Channels
    // [0, 1, 2, 3] correspond to the [R, G, B, A] colors.
    void writeChannel(const Channel& data, const size_t channel);

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
