cmake_minimum_required(VERSION 3.28)
project(week02_tiled_gemm LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

# === Library (kernels + dispatcher) ===
add_library(week02_lib STATIC
    src/gemm_naive.cu
    src/gemm_tiled_32.cu
    src/gemm_tiled_64.cu
    src/gemm_tiled_async.cu
    src/gemm_launch.cu
)
target_include_directories(week02_lib PUBLIC src)
target_link_libraries(week02_lib PUBLIC CUDA::cudart CUDA::cublas)

# === Tests ===
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE week02_lib GTest::gtest_main CUDA::cublas)
add_test(NAME test_gemm COMMAND test_gemm)

# === Bench ===
add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE week02_lib CUDA::cublas)
