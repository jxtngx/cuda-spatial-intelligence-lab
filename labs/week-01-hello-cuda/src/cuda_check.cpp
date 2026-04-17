#include "cuda_check.hpp"

namespace cudalab {
void check_last_error(const char* what) {
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string{what} + " failed: " +
                                 cudaGetErrorString(e));
    }
}
}
