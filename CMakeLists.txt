cmake_minimum_required(VERSION 3.15)

project(DPP-Rasterization)

list(APPEND CMAKE_PREFIX_PATH "/home/makanib/cdux/viskores-install")
find_package(Viskores REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust DEVICE OMP)



