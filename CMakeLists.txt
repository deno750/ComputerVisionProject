cmake_minimum_required(VERSION 2.8)

project(ComputerVision)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(
    Project
    src/main.cpp
    src/feature_extraction.h
    src/image_prediction.h
)

target_link_libraries(Project ${OpenCV_LIBS})
