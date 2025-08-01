cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_COMPILER "/packages/cuda/11.5.1/bin/nvcc" CACHE PATH "" FORCE)

project(DPP-Rasterization LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust HOST CPP DEVICE OMP)

file(GLOB SRC_FILES *.cu)

add_executable(rast ${SRC_FILES})
target_link_libraries(rast Thrust)
target_include_directories(rast PRIVATE .)
