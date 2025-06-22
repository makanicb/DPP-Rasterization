cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_COMPILER "/packages/cuda/11.5.1/bin/nvcc" CACHE PATH "" FORCE)

project(DPP-Rasterization LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

list(APPEND CMAKE_PREFIX_PATH "/home/makanib/cdux/viskores-install")
find_package(Viskores REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust HOST CPP DEVICE OMP)

file(GLOB SRC_FILES *.cu)

add_executable(rast ${SRC_FILES})
target_include_directories(rast PRIVATE .)
target_link_libraries(rast Thrust viskores::cont viskores::worklet)
target_compile_options(rast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
