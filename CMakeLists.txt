cmake_minimum_required(VERSION 3.15)

# Create the project
project(DPP-Rasterization LANGUAGES CXX CUDA)

# Cache vairables for compile options and definitions
option(DEBUG_MODE "Enable debug mode" ON)
set(OPTIMIZATION_LEVEL "O0" CACHE STRING "Optimization level (O0, O1, O2, O3)")
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
set(DEBUG_LEVEL "0" CACHE STRING "Debug level (0-4)")
set(TIME_LEVEL "0" CACHE STRING "Time level (0-2)")

# Set compile options
set_property(CACHE OPTIMIZATION_LEVEL PROPERTY STRINGS "O0" "O1" "O2" "O3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "4")
set_property(CACHE TIME_LEVEL PROPERTY STRINGS "0" "1" "2")

# Set target GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find source files
file(GLOB SRC_FILES *.cu)

# Create the executable
add_executable(rast ${SRC_FILES})

# Include local libraries
target_include_directories(rast PRIVATE .)

# Build compile options
SET(CXX_COMPILE_OPTIONS "")
SET(CUDA_COMPILE_OPTIONS "")

# Debug symbols
if(DEBUG_MODE)
	list(APPEND CXX_COMPILE_OPTIONS "-g")
	list(APPEND CUDA_COMPILE_OPTIONS "-g" "-G")
endif()

# Optimization level
list(APPEND CXX_COMPILE_OPTIONS "-${OPTIMIZATION_LEVEL}")
list(APPEND CUDA_COMPILE_OPTIONS "-${OPTIMIZATION_LEVEL}")

# Wanrings
if(ENABLE_WARNINGS)
	list(APPEND CXX_COMPILE_OPTIONS "-W")
	list(APPEND CUDA_COMPILE_OPTIONS "-W")
endif()

# Apply compile options
target_compile_options(rast PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>
	$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

# Debug and Time level
if(DEBUG_MODE)
	target_compile_definitions(rast PRIVATE DEBUG=${DEBUG_LEVEL} TIME=${TIME_LEVEL})
else()
	target_compile_definitions(rast PRIVATE TIME=${TIME_LEVEL})
endif()
