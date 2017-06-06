#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

#include "Arguments.h"
#include "Image.h"
#include "FileIO.h"
#include "HoughTransform.h"

using std::vector;
using std::thread;
using std::cout;
using std::endl;

// Returns the largest element in the given 2D array.
template <class T>
T getMax(T* buffer, size_t width, size_t height)
{
    T max = buffer[0];
    for (size_t i = 0; i < width * height; ++i)
    {
        if (buffer[i] > max)
            max = buffer[i];
    }

    return max;
}

// Combines all of the channels in 'buffers' to form a single result. The result
// is normalized to the range [0, 1] and gamma correction is applied.
template <class T>
Channel<T> merge(const vector<Channel<T>>& buffers, const Arguments<T>& args)
{
    const size_t outWidth  = args.destImageWidth;
    const size_t outHeight = args.destImageHeight;

    Channel<T> out;
    out.data   = new T[outWidth * outHeight]();
    out.width  = outWidth;
    out.height = outHeight;
    out.stride = 1;

    for (size_t i = 0; i < buffers.size(); ++i)
    {
        for (size_t j = 0; j < outWidth * outHeight; ++j)
            out.data[j] += buffers[i].data[j];

        delete[] buffers[i].data;
    }

    // Scale all pixels to the range [0, 1], and then apply gamma correction
    T max = getMax(out.data, outWidth, outHeight);
    for (size_t i = 0; i < outWidth * outHeight; ++i)
    {
        T val       = out.data[i] / max;
        out.data[i] = pow(val, T{1.0} / args.gamma);
    }
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
    Channel<T> in = input.readChannel<T>(channel);
    HoughTransform<T> hough(args.minThetaDegrees, args.maxThetaDegrees, output.getWidth());
    vector<Channel<T>> buffers(args.numThreads);
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

        // Set up the accumulation buffer for this piece
        buffers[i].data   = new T[outWidth * outHeight]();
        buffers[i].width  = outWidth;
        buffers[i].height = outHeight;
        buffers[i].stride = 1;

        // Start working
        threads.emplace_back
        (
            &HoughTransform<T>::transform, &hough,
            std::cref(in), window, args.minContrast, std::ref(buffers[i])
        );
    }

    for (size_t i = 0; i < threads.size(); ++i)
        threads[i].join();

    // Merge the buffers from each thread and write to the output image.
    Channel<T> out = merge(buffers, args);
    output.writeChannel(out, channel);
    delete[] out.data;
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
    Arguments<double> args;
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
