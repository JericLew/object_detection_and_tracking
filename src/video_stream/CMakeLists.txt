cmake_minimum_required(VERSION 2.8)
project( video_stream2 )
# Set to your opencv directory
set(OpenCV_DIR /home/jeric/opencv-4.5.4-linux/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( video_stream2 video_stream2.cpp )
target_link_libraries( video_stream2 ${OpenCV_LIBS} )