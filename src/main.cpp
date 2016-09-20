#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>
#include <chrono>
#include "Image.h"
#include "FileIO.h"

using namespace std;
using namespace std::chrono;

const double PI = 3.1415926535;

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
    double minThetaDegrees;
    double maxThetaDegrees;
    size_t minY;
    size_t maxY;
    size_t minContrast;
    double gamma;

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
        minContrast(64),
        gamma(1.0),
        numThreads(0)
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
            args.minContrast = atoi(argv[i + 1]);
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
    
    return true;
}

// Returns the largest element in the given 2D array.
double getMax(double* buffer, size_t width, size_t height)
{
    double max = 0;
    for (size_t i = 0; i < width * height; ++i)
    {
        if (buffer[i] > max)
            max = buffer[i];
    }

    return max;
}

// Helper for 'accumulate'. Returns true when (centerX, centerY) is brighter than
// its neighbors.
bool highContrast(Image& input, size_t channel, int centerX, int centerY, int minContrast)
{
    byte* pixels  = input.getData();
    size_t width  = input.getWidth();
    size_t height = input.getHeight();

    signed char centerValue = pixels[4 * (centerY * width + centerX) + channel];

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
            if (abs((signed char)pixels[index] - centerValue) >= minContrast)
                return true;
        }
    }

    return false;
}

// This function is called from inside one of the helper threads. It performs
// the actual computations for the Hough transform.
void accumulate(Image& input, double* accumulationBuffer, size_t channel, Arguments& args, 
                size_t minY, size_t maxY, double* sinTable, double* cosTable)
{
    const size_t inWidth       = input.getWidth();
    const size_t inHeight      = input.getHeight();
    const size_t outWidth      = args.destImageWidth;
    const size_t outHeight     = args.destImageHeight;
    const size_t maxRadius     = (int)ceil(sqrt(inWidth * inWidth + inHeight * inHeight));
    const size_t halfRAxisSize = outHeight / 2;

    for (size_t y = minY; y <= maxY; ++y)
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
                    // To avoid discretization artifacts, we add a bit to both
                    // cells covered by val. The weight is proportional to how
                    // close we are to the lower bound or the upper bound.
                    double val  = (r * halfRAxisSize / maxRadius) + halfRAxisSize;
                    int rLow    = (int) floor(val);
                    int rHi     = (int) ceil(val);
                    
                    double weightHigh = val - (int) val;
                    double weightLow  = 1.0 - weightHigh;
                    
                    // Update the corresponding slot in the accumulator buffer
                    int indexLow  = rLow * outWidth + theta;
                    int indexHigh = rHi  * outWidth + theta;
                    accumulationBuffer[indexLow]  += weightLow;
                    accumulationBuffer[indexHigh] += weightHigh;
                }
            }
        }
    }
}

// Calculates the Hough transform of the given input image on the given channel.
// The results will be placed in the corresponding channel in the output image.
void houghTransform(Image& input, Image& output, size_t channel, Arguments& args)
{
    const size_t outWidth  = output.getWidth();
    const size_t outHeight = output.getHeight();

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

    // Calculate the actual transform in multiple threads
    vector<double*> buffers;
    vector<thread> threads;
    for (size_t i = 0; i < args.numThreads; ++i)
    {
        // Divide the working area into segments for each thread
        size_t inHeight = input.getHeight();
        size_t minY     = i * (inHeight / args.numThreads);
        size_t maxY     = (i + 1) * (inHeight / args.numThreads) - 1;
        if (i == args.numThreads - 1)
            maxY = inHeight - 1;

        buffers.push_back(new double[outWidth * outHeight]());
        threads.push_back(thread(accumulate, std::ref(input), buffers[i], 
            channel, std::ref(args), minY, maxY, sinTable, cosTable));
    }

    for (size_t i = 0; i < args.numThreads; ++i)
        threads[i].join();
    
    // Combine the various buffers into one
    double* accumulationBuffer = new double[outWidth * outHeight]();
    for (size_t i = 0; i < args.numThreads; ++i)
    {
        for (size_t j = 0; j < outWidth * outHeight; ++j)
            accumulationBuffer[j] += buffers[i][j];
    }

    // Determine the brightest element in this channel
    double max = getMax(accumulationBuffer, outWidth, outHeight);

    // Convert the accumulation to a valid channel by converting each cell
    // to a corresponding grey level. Ignore the other channels
    byte* outputPixels = output.getData();
    for (size_t i = 0; i < outWidth * outHeight; ++i)
	{
        // Scale the pixels to [0, 1], and then apply gamma correction
        double val       = accumulationBuffer[i] / max;
        double corrected = pow(val, 1.0 / args.gamma);
        outputPixels[4 * i + channel] = (byte) (corrected * 255);
    }

    delete[] sinTable;
    delete[] cosTable;
    delete[] accumulationBuffer;
    for (size_t i = 0; i < args.numThreads; ++i)
        delete[] buffers[i];
}

// Go through the work necessary to process a single file. We load the image
// into memory, perform the Hough Transform, and write the result to the given
// output file.
void processFile(const string& inputFilename, const string& outputFilename, 
    Arguments& args)
{
    // Load the input image
    Image input;
    if (!input.load(inputFilename))
    {
        cerr << "Unable to open file: " << inputFilename << endl;
        return;
    }

    // Some arguments are resolved at runtime.
    if (args.maxY == 0)
        args.maxY = input.getHeight() - 1;
    if (args.numThreads == 0)
        args.numThreads = thread::hardware_concurrency();

    // Create the output image
    Image output(args.destImageWidth, args.destImageHeight);

    if (args.srcChannel == -1)
    {
        for (size_t i = 0; i < 3; ++i)
            houghTransform(input, output, i, args);
    }
    else houghTransform(input, output, args.srcChannel, args);

    // Save the output image
    if (!output.save(outputFilename))
        cerr << "Unable to open file: " << outputFilename << endl;

    else cout << outputFilename << " successfully saved!" << endl;
}

int main(int argc, char** argv)
{
    auto startTime = high_resolution_clock::now();
    
    // Sort out the command line arguments
    Arguments args;
    if (!parseArguments(argc, argv, args))
        return 1;

    // Initialize DevIL
    ilInit();
    ilSetInteger(IL_JPG_QUALITY, 99);

    // Input is a directory of images
    if (args.useDirectory)
    {
        vector<string> filenames = listFiles(args.srcImageFilename);
        for (int i = 0; i < filenames.size(); ++i)
        {
            string inputFilename  = args.srcImageFilename  + "/" + filenames[i];
            string outputFilename = args.destImageFilename + "/" + filenames[i];
            processFile(inputFilename, outputFilename, args);
        }
    }
    
    // Input is a single image
    else processFile(args.srcImageFilename, args.destImageFilename, args);
        
    // Print the elapsed time
    auto endTime = high_resolution_clock::now();
    cout << "Time elapsed (ms): " 
        << duration_cast<milliseconds>(endTime-startTime).count() << endl;
    
    return 0;
}
