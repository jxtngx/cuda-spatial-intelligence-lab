cmake_minimum_required(VERSION 3.28)
project(week01_hello_cuda LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

# === Library ===
add_library(week01_lib STATIC
    src/cuda_check.cpp
    src/stream.cu
    src/axpy.cu
)
target_include_directories(week01_lib PUBLIC src)
target_link_libraries(week01_lib PUBLIC CUDA::cudart)

# === Tests ===
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_axpy tests/test_axpy.cpp)
target_link_libraries(test_axpy PRIVATE week01_lib GTest::gtest_main)
add_test(NAME test_axpy COMMAND test_axpy)

# === Bench ===
add_executable(bench_axpy bench/bench_axpy.cpp)
target_link_libraries(bench_axpy PRIVATE week01_lib)
