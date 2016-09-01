HoughArt is a program designed to create artwork using the Hough Transform. Given a source image, it calculates the transform per channel and writes the results to an output file. Several options are included for customizing the results.

Compilation:
  There are a few options for compilation:
  
  1) Using CMake. The basic usage is:
  
  cd [projectRoot]/build
  cmake [-g ...] ..
  make
  
  This will produce a binary (hough_art) in the 'build' directory.
  
  2) Manual compilation. The source files are in [projectRoot/src]. The associated header files are in [projectRoot/include]. The user must make sure that DevIL is properly linked.

  3) A Netbeans project is also included, which has been set up to simplify development (including calling cmake).
  
Dependencies:
  - DevIL - used for loading/saving images
  - cmake - (optional) used for generating platform-independent makefiles
  
Documentation:
  The first argument is required. It is the name/location of the file to be used for input. All other arguments have both a command and an associated argument of their own. The options are as follows:
  
  -o : Specify the name and location of the output file. DevIL supports many different extensions, including .png and .jpg. The default value is "../images/output.png".
  
  -w : Specify the width of the output image in pixels. The default value is 320.
  
  -h : Specify the height of the output image in pixels. The default value is 160.
  
  -c : Specify which channel of the original input image is used in the Hough calculation. c = 0 corresponds to the red channel, c = 1 corresponds to the green channel, and c = 2 cooresponds to the blue channel. If c = -1, all channels will be used, and the results will be superimposed upon one another.
  
  -C : Specify the contrast (values range from 1 to 255). The larger the value, the fewer "lines" will be kept by the Hough algorithm, resulting in a darker image overall. See the examples in images/contrast for some example images.
  
  -m : Specify the minimum angle (in degrees) used in the algorithm. The default value is 0.0.
  
  -M : Specify the maximum angle (in degrees) used in the algorithm. The default value is 180.0.
  
  -y : Specify the minimum y coordinate of the original image. This is used to effectively crop the transform. The default value is 0.
  
  -Y : Specify the maximum y coordinate of the original image. This is used to effectively crop the transform. The default value is 0, signifying that the entire image should be used.
  
Example usage:
	houghart input.jpg -o output.png -w 1024 -h 768 -C 64
	