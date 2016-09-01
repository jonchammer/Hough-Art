#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <cstring>
#include "il.h"
using namespace std;

typedef unsigned char byte;

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