#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include "Image.h"
using namespace std;

struct Arguments
{
	string srcImageFilename;
	string destImageFilename;	
};

bool parseArguments(int argc, char** argv, Arguments& args)
{
	if (argc < 2)
	{
		cerr << "Usage: houghart input.jpg [-o output.png]" << endl;
		return false;
	}
	
	args.srcImageFilename = argv[1];
	
	for (int i = 2; i < argc; ++i)
	{
		if (strcmp(argv[i], "-o"))
		{
			args.destImageFilename = argv[i];
			++i;
		}
	}
	
	// Set some default values for the arguments
	if (args.destImageFilename == "")
		args.destImageFilename = "../images/output.png";
	
	return true;
}

bool highContrast(Image& input, size_t channel, int x, int y, int minContrast)
{
	byte* pixels  = input.getData();
	size_t width  = input.getWidth();
	size_t height = input.getHeight();
	
	byte centerValue = pixels[4 * (y * width + x) + channel];
	if (centerValue == 0xFF) return false;
	
	for (int i = 8; i >= 0; --i)
	{
		if (i == 4)
			continue;
		
		int newX = x + (i % 3) - 1;
		int newY = y + (i / 3) - 1;
		if (newX < 0 || newX >= (int) width || newY < 0 || newY >= (int) height)
			continue;
		
		if (abs((pixels[4 * (newY * width + newX) + channel]) - centerValue) >= minContrast)
			return true;
	}
	
	return false;
}
	
void houghTransform(Image& input, Image& output, size_t channel,
	double minTheta, double maxTheta, size_t thetaLevels, size_t radiusLevels, 
	size_t minY, size_t maxY, size_t minContrast)
{
	size_t width  = input.getWidth();
	size_t height = input.getHeight();
	
	size_t oWidth      = output.getWidth();
	size_t oHeight     = output.getHeight();
	
	size_t* accumulationBuffer = new size_t[oWidth * oHeight];
	byte* outputPixels = output.getData();
	
	size_t maxRadius     = (int)ceil(sqrt(width * width + height * height));
	size_t halfRAxisSize = radiusLevels >> 1;
	
	// X output ranges from minTheta to maxTheta degrees
	// Y output ranges from -maxRadius to maxRadius
	const double DEGREES_TO_RADIANS = 3.1415926535 / 180.0;
	const double range              = maxTheta - minTheta;
	
	// Precompute the sin and cos values we will use for each value of theta
	double* sinTable = new double[thetaLevels];
	double* cosTable = new double[thetaLevels];
	for (size_t theta = 0; theta < thetaLevels; ++theta)
	{
		double thetaDegrees = (theta * range) / thetaLevels + minTheta;
		double thetaRadians = thetaDegrees * DEGREES_TO_RADIANS;
		sinTable[theta]     = sin(thetaRadians);
		cosTable[theta]     = cos(thetaRadians);
	}
	
	// Perform the accumulation
	size_t max = 0;

	for (size_t y = minY; y <= maxY; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			if (highContrast(input, channel, x, y, minContrast))
			{
				for (size_t theta = 0; theta < thetaLevels; ++theta)
				{
					// Calculate r using polar normal form (r = x*cos(theta) + y*sin(theta))
					double r    = cosTable[theta] * x + sinTable[theta] * y;
					
					// Scale r to the range [-maxRadius/2, maxRadius]
					int rScaled = (int) round(r * halfRAxisSize / maxRadius) + halfRAxisSize;
					
					// Update the corresponding slot in the accumulator buffer
					int index      = rScaled * oWidth + theta;
					size_t current = accumulationBuffer[index];
					accumulationBuffer[index]++;

					// Update the max value
					if (current > max)
						max = current + 1;
				}
			}
		}
	}
	
	// Convert the accumulation to a valid channel by converting each cell
	// to a corresponding grey level
	for (size_t i = 0; i < oWidth * oHeight; ++i)
	{
		byte value = (byte) (((double) accumulationBuffer[i] / (double) max) * 255);
		outputPixels[4 * i + 0] = value;
		outputPixels[4 * i + 1] = value;
		outputPixels[4 * i + 2] = value;
		outputPixels[4 * i + 3] = 0xFF;
		//if (outputPixels[4 * i + channel] != 0)
		//cout << "i: " << i << " - " << (int) outputPixels[4 * i + channel] << endl;
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
	
	cout << "INPUT: " << args.srcImageFilename << endl;
	cout << "OUTPUT:" << args.destImageFilename << endl;
	// Initialize DevIL
	ilInit();
	
	// Load the input image
	Image input;
	if (!input.load(args.srcImageFilename))
	{
		cerr << "Unable to open file: " << args.srcImageFilename << endl;
		return 2;
	}
	
	size_t channel      = 0;
	double minTheta     = 0.0;
	double maxTheta     = 3.1415926535;
	size_t thetaLevels  = 320;
	size_t radiusLevels = 256;
	size_t minY         = 0;
	size_t maxY         = input.getHeight() - 1;
	size_t minContrast  = 64;
	
	Image output(thetaLevels, radiusLevels);
	houghTransform(input, output, channel, minTheta, maxTheta, thetaLevels, radiusLevels, minY, maxY, minContrast);
	
	if (!output.save(args.destImageFilename))
	{
		cerr << "Unable to open file: " << args.destImageFilename << endl;
		return 3;
	}
	else cout << args.destImageFilename << " successfully saved!" << endl;

	return 0;
}