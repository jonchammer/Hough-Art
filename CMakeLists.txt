cmake_minimum_required(VERSION 3.8)
project(hough_art)

# If the user doesn't specify a build type (e.g. with -DCMAKE_BUILD_TYPE=Release),
# choose Release as the default value.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Building in ${CMAKE_BUILD_TYPE} mode.")

# Define the sources
set(SOURCES
    src/Image.cpp
    src/main.cpp
)

# Define the main executable
set(DEBUG_OPTIONS -g)
set(RELEASE_OPTIONS -O3)
add_executable(hough_art ${SOURCES})
target_compile_features(hough_art PRIVATE cxx_std_14)
target_include_directories(hough_art
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(hough_art PRIVATE -Wall -Wextra -pthread)
target_compile_options(hough_art PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(hough_art PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

# Attach threading library
find_package(Threads REQUIRED)
target_link_libraries(hough_art PRIVATE Threads::Threads)

# Attach DevIL
find_package(DevIL REQUIRED)
target_include_directories(hough_art PRIVATE ${IL_INCLUDE_DIR})
target_link_libraries(hough_art PRIVATE ${IL_LIBRARIES})

# Copy all the DLLs from 'libs' into the executable directory
if (WINDOWS)
    file(GLOB DLLS "libs/*.dll")
    file(COPY ${DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
endif()
