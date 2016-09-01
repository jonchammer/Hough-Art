#include "Image.h"

Image::Image()
    : mWidth(1), mHeight(1), mData(NULL), mOwnData(false)
{
    ilGenImages(1, &mID);
}

Image::Image(size_t width, size_t height)
    : mWidth(width), mHeight(height), mOwnData(true)
{
    ilGenImages(1, &mID);
    ilBindImage(mID);

    // Create a buffer we can work with, and bind it to a given ID 
    // so DevIL can work with it
    mData = new byte[width * height * 4]();
    
    // Initialize the alpha channel to 0xFF
    for (size_t i = 0; i < width * height; ++i)
        mData[4 * i + 3] = 0xFF;
    
    ilTexImage(width, height, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, mData);
}

Image::Image(const Image& other)
{
    ilGenImages(1, &mID);
    ilBindImage(mID);

    mWidth  = other.mWidth;
    mHeight = other.mHeight;

    // Create a new byte buffer for this image and copy the data
    // from the original
    mData    = new byte[mWidth * mHeight * 4];
    mOwnData = true;
    memcpy(mData, other.mData, mWidth * mHeight * 4);
    ilTexImage(mWidth, mHeight, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, mData);
}

Image::~Image()
{
    // If we own the data, we need to deallocate it
    if (mOwnData) delete[] mData;

    ilDeleteImages(1, &mID);
}

bool Image::load(const string& filename)
{
    ilBindImage(mID);
    if (!ilLoadImage(filename.c_str()))
        return false;

    // Save the width and height 
    mWidth  = ilGetInteger(IL_IMAGE_WIDTH);
    mHeight = ilGetInteger(IL_IMAGE_HEIGHT);

    // Convert all incoming images to a common format
    // that we can work with (RGBA).
    ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);
    mData    = ilGetData();
    mOwnData = false;

    return true;
}

bool Image::save(const string& filename)
{
    ilBindImage(mID);
    ilEnable(IL_FILE_OVERWRITE);
    
    // We have to rebind the data so DevIL is kept in sync
    ilTexImage(mWidth, mHeight, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, mData);
    return ilSaveImage(filename.c_str());
}

size_t Image::getWidth() const
{
    return mWidth;
}

size_t Image::getHeight() const
{
    return mHeight;
}

const byte* Image::getData() const
{
    return mData;
}

byte* Image::getData()
{
    return mData;
}
	
	