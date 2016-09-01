#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include "Image.h"
using namespace std;

const double PI = 3.1415926535;

struct Arguments
{
    string srcImageFilename;
    int srcChannel;
    
    string destImageFilename;
    size_t destImageWidth;
    size_t destImageHeight;
    
    double minThetaDegrees;
    double maxThetaDegrees;
    size_t minY;
    size_t maxY;
    size_t minContrast;
    
    // Assign default values for each argument
    Arguments() : 
        srcChannel(-1),
        destImageFilename("../images/output.png"), 
        destImageWidth(320), 
        destImageHeight(160),
        minThetaDegrees(0.0),
        maxThetaDegrees(180.0),
        minY(0),
        maxY(0),
        minContrast(64)
    {}
};

bool parseArguments(int argc, char** argv, Arguments& args)
{
    if (argc < 2)
    {
        cerr << "Usage: houghart input.jpg [-o output.png] [-w 1024] [-h 768]" << endl;
        return false;
    }

    args.srcImageFilename = argv[1];

    for (int i = 2; i < argc; ++i)
    {
        if (strcmp(argv[i], "-o") == 0)
        {
            args.destImageFilename = argv[i + 1];
            ++i;
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
    }
    
    return true;
}

bool highContrast(Image& input, size_t channel, int centerX, int centerY, int minContrast)
{
    byte* pixels     = input.getData();
    size_t width     = input.getWidth();
    size_t height    = input.getHeight();
    char centerValue = pixels[4 * (centerY * width + centerX) + channel];

    for (int y = -1; y <= 1; ++y)
    {
        int newY = centerY + y;
        if (newY < 0 || newY >= (int) height)
            continue;
        
        for (int x = -1; x <= 1; ++x)
        {
            int newX = centerX + x;
            if (newX < 0 || newX >= (int) width)
                continue;
            
            int index = 4 * (newY * width + newX) + channel;
            if (abs((char)pixels[index] - centerValue) >= minContrast)
                return true;
        }
    }
    
    return false;
}
	
void houghTransform(Image& input, Image& output, size_t channel, Arguments& args)
{
    const size_t inWidth   = input.getWidth();
    const size_t inHeight  = input.getHeight();
    const size_t outWidth  = output.getWidth();
    const size_t outHeight = output.getHeight();

    const size_t maxRadius     = (int)ceil(sqrt(inWidth * inWidth + inHeight * inHeight));
    const size_t halfRAxisSize = outHeight / 2;

    // X output ranges from minTheta to maxTheta degrees
    // Y output ranges from -maxRadius to maxRadius
    const double minThetaRadians = PI / 180.0 * args.minThetaDegrees;
    const double maxThetaRadians = PI / 180.0 * args.maxThetaDegrees;
    const double range           = maxThetaRadians - minThetaRadians;

    // Precompute the sin and cos values we will use for each value of theta
    double* sinTable = new double[outWidth];
    double* cosTable = new double[outWidth];
    for (size_t theta = 0; theta < outWidth; ++theta)
    {
        // Map theta from [0, width] -> [minThetaRadians, maxThetaRadians]
        double thetaRadians = (theta * range) / outWidth + minThetaRadians;
        sinTable[theta]     = sin(thetaRadians);
        cosTable[theta]     = cos(thetaRadians);
    }

    // Perform the accumulation
    size_t max = 0;
    size_t* accumulationBuffer = new size_t[outWidth * outHeight];
    for (size_t y = args.minY; y <= args.maxY; ++y)
    {
        for (size_t x = 0; x < inWidth; ++x)
        {
            // Only process pixels that are brighter than its surroundings
            if (highContrast(input, channel, x, y, args.minContrast))
            {
                for (size_t theta = 0; theta < outWidth; ++theta)
                {
                    // Calculate r using the polar normal form: 
                    // (r = x*cos(theta) + y*sin(theta))
                    double r = cosTable[theta] * x + sinTable[theta] * y;

                    // Scale r to the range [-maxRadius/2, maxRadius/2]
                    int rScaled = (int) round(r * halfRAxisSize / maxRadius) 
                        + halfRAxisSize;

                    // Update the corresponding slot in the accumulator buffer
                    int index      = rScaled * outWidth + theta;
                    size_t current = accumulationBuffer[index];
                    accumulationBuffer[index]++;

                    // Update the max value as we go
                    if (current > max)
                        max = current + 1;
                }
            }
        }
    }

    // Convert the accumulation to a valid channel by converting each cell
    // to a corresponding grey level. Ignore the other channels
    byte* outputPixels = output.getData();
    for (size_t i = 0; i < outWidth * outHeight; ++i)
    {
        outputPixels[4 * i + channel] = 
            (byte) (((double) accumulationBuffer[i] / (double) max) * 255);
    }

    delete[] sinTable;
    delete[] cosTable;
    delete[] accumulationBuffer;
}

int main(int argc, char** argv)
{
    // Sort out the command line arguments
    Arguments args;
    if (!parseArguments(argc, argv, args))
        return 1;

    // Initialize DevIL
    ilInit();

    // Load the input image
    Image input;
    if (!input.load(args.srcImageFilename))
    {
        cerr << "Unable to open file: " << args.srcImageFilename << endl;
        return 2;
    }

    if (args.maxY == 0)
        args.maxY = input.getHeight() - 1;
    
    // Create the output image
    Image output(args.destImageWidth, args.destImageHeight);
    
    if (args.srcChannel == -1)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            houghTransform(input, output, i, args);
        }
    }
    else houghTransform(input, output, args.srcChannel, args);

    // Save the output image
    if (!output.save(args.destImageFilename))
    {
        cerr << "Unable to open file: " << args.destImageFilename << endl;
        return 3;
    }
    else cout << args.destImageFilename << " successfully saved!" << endl;

    return 0;
}