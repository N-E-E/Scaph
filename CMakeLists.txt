cmake_minimum_required(VERSION 3.10)

project(scaph LANGUAGES CXX CUDA)

find_package(gflags REQUIRED)
find_package(CUDA 11.6 REQUIRED)

include_directories(tools)
include_directories(src)
add_subdirectory(app)
