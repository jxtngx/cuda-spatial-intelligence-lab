cmake_minimum_required(VERSION 3.28)
project(lab LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

# === Library (kernels + host wrappers) ===
add_library(lab_lib STATIC
    src/placeholder.cu
)
target_include_directories(lab_lib PUBLIC src)
target_link_libraries(lab_lib PUBLIC CUDA::cudart CUDA::nvtx3)

# === Tests ===
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_placeholder tests/test_placeholder.cpp)
target_link_libraries(test_placeholder PRIVATE lab_lib GTest::gtest_main)
add_test(NAME test_placeholder COMMAND test_placeholder)

# === Benchmarks ===
add_executable(bench_placeholder bench/bench_placeholder.cpp)
target_link_libraries(bench_placeholder PRIVATE lab_lib)
