cmake_minimum_required(VERSION 3.24)

project(KinectReader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV root directory, without it find_package() was failing
set(OpenCV_DIR
    "C:/opencv/build/x64/vc16/lib"
)
find_package(OpenCV REQUIRED)

# Kinect for Windows SDK 2.0 root directory using the environment variable KINECTSDK20_DIR, then perform an error check
set(KINECT_SDK_ROOT "$ENV{KINECTSDK20_DIR}")

if(NOT KINECT_SDK_ROOT)
    message(FATAL_ERROR "KINECTSDK20_DIR is not defined.")
endif()

add_executable(
    KinectReader
    src/main.cpp
)

# Give the compiler access to OpenCV and Kinect SDK header files.
target_include_directories(
    KinectReader
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        "${KINECT_SDK_ROOT}/inc"
)

# Tell the linker where Kinect20.lib is located, then link the libraries
target_link_directories(
    KinectReader
    PRIVATE
        "${KINECT_SDK_ROOT}/Lib/x64"
)

target_link_libraries(
    KinectReader
    PRIVATE

    # OpenCV libraries
    ${OpenCV_LIBS}

    # Kinect for Windows SDK 2.0
    Kinect20.lib

    # Windows Sockets 2 networking library
    Ws2_32.lib
)