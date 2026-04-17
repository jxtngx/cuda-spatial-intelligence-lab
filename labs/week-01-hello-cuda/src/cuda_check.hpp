#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string{#expr} + " failed: " +        \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

namespace cudalab {
void check_last_error(const char* what);
}
