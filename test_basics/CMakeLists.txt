cmake_minimum_required(VERSION 3.15)

set(TEST_EXP "8" CACHE STRING "The log base 2 of the array size")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(TestAddArrays LANGUAGES CXX CUDA)

add_executable(test test_add_arrays.cu)

set(my_compile_options "-O3")
target_compile_options(test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_compile_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_compile_options}>
)
target_compile_definitions(test PRIVATE EXP=${TEST_EXP})
