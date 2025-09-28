cmake_minimum_required(VERSION 3.13)

set(TEST_EXP "8" CACHE STRING "Log of size of arrays")

project (TestAddArray CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

list(APPEND CMAKE_PREFIX_PATH "/global/common/software/m2956/viskores")
find_package(Viskores REQUIRED)

add_executable(test test_add_array.cu)
target_link_libraries(test viskores::cont)
set(my_comp_options "-g -O3")
target_compile_options(test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_comp_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_comp_options}>
)
target_compile_definitions(test PRIVATE EXP=${TEST_EXP})
