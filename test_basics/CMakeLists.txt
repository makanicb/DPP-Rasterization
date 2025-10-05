cmake_minimum_required(VERSION 3.15)

set(TEST_EXP "8" CACHE STRING "The log base 2 of the array size")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(TestAddArrays LANGUAGES CXX CUDA)

add_executable(test_add test_add_arrays.cu)

target_compile_options(test_add PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_compile_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_compile_options}>
)
target_compile_definitions(test_add PRIVATE EXP=${TEST_EXP})

add_executable(test_work test_functor.cu)

target_compile_options(test_work PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_compile_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_compile_options}>
)
target_compile_definitions(test_work PRIVATE EXP=${TEST_EXP})
