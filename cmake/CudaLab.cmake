# Shared CMake helpers for cuda-spatial-intelligence-lab labs.
#
# Sets the standard target architecture (Blackwell sm_121 on DGX Spark),
# language standards (C++20 / CUDA C++20), warning levels, and common
# CUDA flags. Use from any lab's CMakeLists.txt:
#
#   include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
#   cuda_lab_defaults()

function(cuda_lab_defaults)
    set(CMAKE_CXX_STANDARD 20 PARENT_SCOPE)
    set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
    set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)

    set(CMAKE_CUDA_STANDARD 20 PARENT_SCOPE)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON PARENT_SCOPE)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)

    # sm_121 = Blackwell on DGX Spark. Override only if a lab explicitly
    # tests portability across architectures.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "121" PARENT_SCOPE)
    endif()

    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release PARENT_SCOPE)
    endif()

    find_package(CUDAToolkit REQUIRED)

    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
endfunction()
