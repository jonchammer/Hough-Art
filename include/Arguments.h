#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include <cstring>
#include <cstdlib>
#include <iostream>

using std::string;
using std::cerr;
using std::endl;

template <class T>
struct Arguments
{
    // Source information
    string srcImageFilename;
    int srcChannel;
    bool useDirectory;

    // Destination information
    string destImageFilename;
    size_t destImageWidth;
    size_t destImageHeight;

    // Transformation parameters
    T minThetaDegrees;
    T maxThetaDegrees;
    size_t minY;
    size_t maxY;
    T minContrast;
    T gamma;

    // Program options
    size_t numThreads;

    // Assign default values for each argument
    Arguments() :
        srcChannel(-1),
        useDirectory(false),
        destImageFilename("../images/output.png"),
        destImageWidth(320),
        destImageHeight(160),
        minThetaDegrees(0.0),
        maxThetaDegrees(180.0),
        minY(0),
        maxY(0),
        minContrast(0.5),
        gamma(1.0),
        numThreads(0)
    {}
};

template <class T> bool parseArguments(int argc, char** argv, Arguments<T>& args)
{
    if (argc < 2)
    {
        cerr << "Usage: houghart input.jpg [-o output.png] [-w 1024] [-h 768]" << endl;
        return false;
    }

    args.srcImageFilename = argv[1];
    bool customOutput     = false;

    for (int i = 2; i < argc; ++i)
    {
        if (strcmp(argv[i], "-o") == 0)
        {
            args.destImageFilename = argv[i + 1];
            ++i;
            customOutput = true;
        }
        else if (strcmp(argv[i], "-w") == 0)
        {
            args.destImageWidth = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-h") == 0)
        {
            args.destImageHeight = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-c") == 0)
        {
            args.srcChannel = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-m") == 0)
        {
            args.minThetaDegrees = atof(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-M") == 0)
        {
            args.maxThetaDegrees = atof(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-y") == 0)
        {
            args.minY = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-Y") == 0)
        {
            args.maxY = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-C") == 0)
        {
            args.minContrast = atof(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-g") == 0)
        {
            args.gamma = atof(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-n") == 0)
        {
            args.numThreads = atoi(argv[i + 1]);
            ++i;
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            args.useDirectory = true;
        }
    }

    // If the user chose to do batch processing, but didn't specify a directory,
    // ensure the default option is actually a directory.
    if (args.useDirectory && customOutput)
        args.destImageFilename = "../images";

    #ifdef ENABLE_CUDA
        args.numThreads = 1;
    #endif
    return true;
}

#endif
