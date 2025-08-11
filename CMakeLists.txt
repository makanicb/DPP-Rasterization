cmake_minimum_required(VERSION 3.15)

# set(CMAKE_CUDA_COMPILER "/packages/cuda/11.5.1/bin/nvcc" CACHE PATH "" FORCE)

project(DPP-Rasterization LANGUAGES CXX CUDA)

# Cache variables for compile options and definitions
option(DEBUG_MODE "Enable debug mode" ON)
set(OPTIMIZATION_LEVEL "O0" CACHE STRING "Optimization level (O0, O1, O2, O3)")
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
set(DEBUG_LEVEL "1" CACHE STRING "Debug Level (0-4)")
set(TIME_LEVEL "1" CACHE STRING "Time Level (0-2)")

# Set optimization level options
set_property(CACHE OPTIMIZATION_LEVEL PROPERTY STRINGS "O0" "O1" "O2" "O3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "4")
set_property(CACHE TIME_LEVEL PROPERTY STRINGS "0" "1" "2")

set(CMAKE_CUDA_ARCHITECTURES 80)

list(APPEND CMAKE_PREFIX_PATH "/home/makanib/cdux/viskores-install")
find_package(Viskores REQUIRED)

# find_package(Thrust REQUIRED CONFIG)
# thrust_create_target(Thrust HOST CPP DEVICE OMP)

# find_package(OpenMP REQUIRED)

#Retrieve source files
file(GLOB SRC_FILES *.cpp *.cu)

add_executable(rast ${SRC_FILES})

# Include local libraries
target_include_directories(rast PRIVATE .)

# Link Viskores
target_link_libraries(rast viskores::cont viskores::worklet)

#Build compile options
set(CXX_COMPILE_OPTIONS "")
set(CUDA_COMPILE_OPTIONS "")

# Debug symbols
if(DEBUG_MODE)
	list(APPEND CXX_COMPILE_OPTIONS "-g")
	list(APPEND CUDA_COMPILE_OPTIONS "-g" "-G")
endif()

# Optimization level
list(APPEND CXX_COMPILE_OPTIONS "-${OPTIMIZATION_LEVEL}")
list(APPEND CUDA_COMPILE_OPTIOSN "-${OPTIMIZATION_LEVEL}")

# Warnings
if(ENABLE_WARNINGS)
	list(APPEND CXX_COMPILE_OPTIONS "-W")
	list(APPEND CUDA_COMPILE_OPTIONS "-W")
endif()

# Apply compile options
target_compile_options(rast PRIVATE 
	$<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

# Define DEBUG level
if(DEBUG_MODE)
	target_compile_definitions(rast PRIVATE DEBUG=${DEBUG_LEVEL} TIME=${TIME_LEVEL})
else()
	target_compile_definitions(rast PRIVATE TIME=${TIME_LEVEL})
endif()
