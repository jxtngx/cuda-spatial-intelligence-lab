cmake_minimum_required(VERSION 3.28)
project(week03_reduce_scan LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

# === Library (kernels) ===
add_library(week03_lib STATIC
    src/reduce.cu
    src/reduce_cub.cu
    src/scan.cu
    src/scan_cub.cu
    src/histogram.cu
)
target_include_directories(week03_lib PUBLIC src)
target_link_libraries(week03_lib PUBLIC CUDA::cudart)

# === Tests ===
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_reduce_scan tests/test_reduce_scan.cpp)
target_link_libraries(test_reduce_scan PRIVATE week03_lib GTest::gtest_main)
add_test(NAME test_reduce_scan COMMAND test_reduce_scan)

# === Bench ===
add_executable(bench_reduce_scan bench/bench_reduce_scan.cpp)
target_link_libraries(bench_reduce_scan PRIVATE week03_lib)
