cmake_minimum_required(VERSION 2.6)
project(hough_art)

# If the user doesn't specify a build type (e.g. with -DCMAKE_BUILD_TYPE=Release),
# choose Release as the default value.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Building in ${CMAKE_BUILD_TYPE} mode.")

# Add compiler flags here (e.g. -O3, -Wall, etc.)
# This is also needed to use C++11. "-std=c++11"
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3")

# Set the location of the final executable to be where cmake/make are called
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Locate DevIL
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${hough_art_SOURCE_DIR}")
find_package(DevIL REQUIRED)
include_directories(${IL_INCLUDE_DIR})

# Locate the proper threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)
if (CMAKE_USE_PTHREADS_INIT)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread")
endif()

# Tell the compiler where to find the headers
include_directories(include)

# All the code is in the src subdirectory
file(GLOB_RECURSE SOURCES "src/*.cpp")

#Ignore the file "GL_Render.cpp"
#list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/GL_Render.cpp")

# The first argument will be the name of the executable (e.g. "[NAME].exe")
add_executable(hough_art ${SOURCES})

# Link with DevIL
target_link_libraries(hough_art ${IL_LIBRARIES})

# Link the thread library
target_link_libraries(hough_art ${CMAKE_THREAD_LIBS_INIT})

# Copy all the DLLs from 'libs' into the executable directory
file(GLOB DLLS "libs/*.dll")
file(COPY ${DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
