cmake_minimum_required(VERSION 3.13)

set(TEST_EXP "8" CACHE STRING "Log of size of arrays")

project (TestAddArray CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

list(APPEND CMAKE_PREFIX_PATH "/global/common/software/m2956/viskores")
find_package(Viskores REQUIRED)

add_executable(test_add test_add_array.cu)
target_link_libraries(test_add viskores::cont)
target_compile_options(test_add PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_comp_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_comp_options}>
)
target_compile_definitions(test_add PRIVATE EXP=${TEST_EXP})

add_executable(test_work test_functor.cu)
target_link_libraries(test_work viskores::cont)
target_compile_options(test_work PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_comp_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_comp_options}>
)
target_compile_definitions(test_work PRIVATE EXP=${TEST_EXP})

add_executable(test_reduce test_reduce.cu)
target_link_libraries(test_reduce viskores::cont)
target_compile_options(test_reduce PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${my_comp_options}>
        $<$<COMPILE_LANGUAGE:CUDA>:${my_comp_options}>
)
