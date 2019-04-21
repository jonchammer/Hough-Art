#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>

#include "Arguments.h"
#include "Image.h"
#include "FileIO.h"
#include "HoughTransform.h"

using std::vector;
using std::thread;
using std::cout;
using std::endl;

// Combines all of the channels in 'buffers' to form a single result. The result
// is normalized to the range [0, 1] and gamma correction is applied.
template <class T>
Channel merge(const vector<Channel>& buffers, const Arguments<T>& args)
{
    const size_t outWidth  = args.destImageWidth;
    const size_t outHeight = args.destImageHeight;

    Channel out(outWidth, outHeight);
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        for (size_t j = 0; j < outWidth * outHeight; ++j)
            out.data[j] += buffers[i].data[j];
    }

    // Scale all pixels to the range [0, 1], and then apply gamma correction
    auto begin    = out.data.begin();
    auto end      = out.data.end();
    const T gamma = args.gamma;
    T max         = *std::max_element(begin, end);
    std::for_each(begin, end, [max, gamma](T& val)
    {
        val = std::pow(val / max, T{1.0} / gamma);
    });
    return out;
}

// Calculates the Hough transform of the given input image on the given channel.
// The results will be placed in the corresponding channel in the output image.
// This function splits the work based on the desired number of threads. It will
// then combine the results into the final product.
template <class T>
void processParallel(const Image& input, Image& output,
    const size_t channel, const Arguments<T>& args)
{
    const size_t outWidth  = output.getWidth();
    const size_t outHeight = output.getHeight();

    // Setup
    Channel in = input.readChannel(channel);
    HoughTransform hough(args.minThetaDegrees, args.maxThetaDegrees, output.getWidth());

    // Create accumulation buffers for each piece
    vector<Channel> buffers;
    for (size_t i = 0; i < args.numThreads; ++i)
        buffers.emplace_back(outWidth, outHeight);

    vector<thread> threads;
    for (size_t i = 0; i < args.numThreads; ++i)
    {
        // Divide the working area into segments for each thread. Each segment
        // gets its own window
        size_t inHeight = input.getHeight();

        Window window;
        window.xMin = 0;
        window.xMax = input.getWidth() - 1;
        window.yMin = i * (inHeight / args.numThreads);
        window.yMax = (i + 1) * (inHeight / args.numThreads) - 1;
        if (i == args.numThreads - 1)
            window.yMax = inHeight - 1;

        // Start working
        threads.emplace_back
        (
            &HoughTransform::transform, &hough,
            std::cref(in), window, args.minContrast, std::ref(buffers[i])
        );
    }

    for (size_t i = 0; i < threads.size(); ++i)
        threads[i].join();

    // Merge the buffers from each thread and write to the output image.
    Channel out = merge(buffers, args);
    output.writeChannel(out, channel);
}

// Go through the work necessary to process a single file. We load the image
// into memory, perform the Hough Transform, and write the result to the given
// output file.
template <class T>
void processFile(const string& inputFilename, const string& outputFilename,
    Arguments<T>& args)
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
            processParallel(input, output, i, args);
    }
    else processParallel(input, output, args.srcChannel, args);

    // Save the output image
    if (!output.save(outputFilename))
        cerr << "Unable to open file: " << outputFilename << endl;

    else cout << outputFilename << " successfully saved!" << endl;
}

int main(int argc, char** argv)
{
    using namespace std::chrono;
    auto startTime = high_resolution_clock::now();

    // Sort out the command line arguments
    Arguments<float> args;
    if (!parseArguments(argc, argv, args))
        return 1;

    // Initialize the image library (if necessary)
    Image::init();

    // Input is a directory of images
    if (args.useDirectory)
    {
        vector<string> filenames = listFiles(args.srcImageFilename);
        for (size_t i = 0; i < filenames.size(); ++i)
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
         << duration_cast<milliseconds>(endTime - startTime).count()
         << endl;

    return 0;
}
