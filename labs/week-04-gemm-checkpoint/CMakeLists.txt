cmake_minimum_required(VERSION 3.28)
project(week04_gemm_checkpoint LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

# === Library (kernels + dispatcher) ===
add_library(week04_lib STATIC
    src/gemm_v4_checkpoint.cu
    src/gemm_launch.cu
)
target_include_directories(week04_lib PUBLIC src)
target_link_libraries(week04_lib PUBLIC CUDA::cudart CUDA::cublas)

# Optional CUTLASS baseline (silently disabled if CUTLASS_DIR not set).
if(DEFINED ENV{CUTLASS_DIR} OR DEFINED CUTLASS_DIR)
    if(NOT DEFINED CUTLASS_DIR)
        set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    endif()
    target_sources(week04_lib PRIVATE src/cutlass_baseline.cu)
    target_include_directories(week04_lib PUBLIC
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    target_compile_definitions(week04_lib PUBLIC CUDALAB_HAVE_CUTLASS=1)
else()
    message(STATUS "week04: CUTLASS_DIR not set, CUTLASS baseline disabled (-1 Performance point).")
endif()

# === Tests ===
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_gemm_checkpoint tests/test_gemm_checkpoint.cpp)
target_link_libraries(test_gemm_checkpoint PRIVATE week04_lib GTest::gtest_main CUDA::cublas)
add_test(NAME test_gemm_checkpoint COMMAND test_gemm_checkpoint)

# === Bench ===
add_executable(bench_gemm_checkpoint bench/bench_gemm_checkpoint.cpp)
target_link_libraries(bench_gemm_checkpoint PRIVATE week04_lib CUDA::cublas)
