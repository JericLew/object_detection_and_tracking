cmake_minimum_required(VERSION 2.8)
project( Tracking )

set(CMAKE_BUILD_TYPE Debug)

# For extra compiler warnings
include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAG"${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

# example usage
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
enable_cxx_compiler_flag_if_supported("-Weffc++")
# enable_cxx_compiler_flag_if_supported("-Wconversion")
# enable_cxx_compiler_flag_if_supported("-Wsign-conversion")

# Set to your opencv directory
set(OpenCV_DIR /home/jeric/opencv-4.5.4-linux/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Tracking Tracking.cpp )
target_link_libraries( Tracking ${OpenCV_LIBS} )
