cmake_minimum_required(VERSION 3.10.0)
project(test_r329)

set(CMAKE_C_FLAGS "-march=armv8-a ")
set(CMAKE_CXX_FLAGS "-march=armv8-a -fpermissive ")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

include_directories(/usr/include/opencv4 include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

aux_source_directory(. src)

add_executable(zhouyi_cam ${src})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

target_link_libraries(zhouyi_cam ${OpenCV_LIBS} aipudrv)

#add_executable(test_fb test_fb.c.bak)


