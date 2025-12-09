cmake_minimum_required(VERSION 3.15)

# set(CMAKE_CUDA_COMPILER "/packages/cuda/11.5.1/bin/nvcc" CACHE PATH "" FORCE)

project(DPP-Rasterization LANGUAGES CXX)

# Cache variables for compile options and definitions
option(DEBUG_MODE "Enable debug mode" ON)
set(OPTIMIZATION_LEVEL "O0" CACHE STRING "Optimization level (O0, O1, O2, O3)")
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
set(DEBUG_LEVEL "1" CACHE STRING "Debug Level (0-4)")
set(TIME_LEVEL "1" CACHE STRING "Time Level (0-3)")
option(CUDA_MODE "Compile for CUDA" OFF)
option(USE_MG_MEM "Use managed memory" OFF)

# Set optimization level options
set_property(CACHE OPTIMIZATION_LEVEL PROPERTY STRINGS "O0" "O1" "O2" "O3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "4")
set_property(CACHE TIME_LEVEL PROPERTY STRINGS "0" "1" "2" "3")

# Turn on CUDA
if(CUDA_MODE)
	enable_language(CUDA) # Enable cuda support

	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CUDA_TARGET_ARCHITECTURE "80" CACHE STRING "CUDA Architecture")
		set(CMAKE_CUDA_ARCHITECTURES ${CUDA_TARGET_ARCHITECTURE})
	endif()
endif()

#list(APPEND CMAKE_PREFIX_PATH "/global/common/software/m2956/viskores") # Computer dependent. Set for your system.
list(APPEND CMAKE_PREFIX_PATH "/home/makanib/cdux/viskores-install") # Computer dependent. Set for your system.
find_package(Viskores REQUIRED)

# find_package(Thrust REQUIRED CONFIG)
# thrust_create_target(Thrust HOST CPP DEVICE OMP)

# find_package(OpenMP REQUIRED)

#Retrieve source files
file(GLOB SRC_FILES *.cpp)

add_executable(rast ${SRC_FILES})

# Compile with CUDA if relevant
if(CUDA_MODE)
	set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CUDA)
endif()

# Include local libraries
target_include_directories(rast PRIVATE .)

# Link Viskores
target_link_libraries(rast viskores::cont viskores::worklet)

#Build compile options
set(CXX_COMPILE_OPTIONS "")
if(CUDA_MODE)
	set(CUDA_COMPILE_OPTIONS "")
endif()

# Debug symbols
if(DEBUG_MODE)
	list(APPEND CXX_COMPILE_OPTIONS "-g")
	if(CUDA_MODE)
		list(APPEND CUDA_COMPILE_OPTIONS "-g" "-G")
	endif()
endif()

# Optimization level
list(APPEND CXX_COMPILE_OPTIONS "-${OPTIMIZATION_LEVEL}")
if(CUDA_MODE)
	list(APPEND CUDA_COMPILE_OPTIOSN "-${OPTIMIZATION_LEVEL}")
endif()

# Warnings
if(ENABLE_WARNINGS)
	list(APPEND CXX_COMPILE_OPTIONS "-W")
	if(CUDA_MODE)
		list(APPEND CUDA_COMPILE_OPTIONS "-W")
	endif()
endif()

# Apply compile options
if(CUDA_MODE)
	target_compile_options(rast PRIVATE 
		$<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>
		$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
	)
else()
	target_compile_options(rast PRIVATE 
		$<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>
	)
endif()

set(CUDA_ON "0")
if(CUDA_MODE)
	set(CUDA_ON "1")
endif()

# Define DEBUG level
if(DEBUG_MODE)
	target_compile_definitions(rast PRIVATE DEBUG=${DEBUG_LEVEL} TIME=${TIME_LEVEL} CUDA_ON=$<BOOL:${CUDA_ON}> MG_MEM=$<BOOL:${USE_MG_MEM}>)
else()
	target_compile_definitions(rast PRIVATE TIME=${TIME_LEVEL} CUDA_ON=$<BOOL:${CUDA_ON}> MG_MEM=$<BOOL:${USE_MG_MEM}>)
endif()
