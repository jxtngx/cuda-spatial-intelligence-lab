#pragma once

#include "cuda_check.hpp"
#include <cuda_runtime.h>
#include <utility>

namespace cudalab {

// Move-only RAII wrapper around cudaStream_t. Non-blocking by default.
class Stream {
public:
    Stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    }
    ~Stream() {
        if (s_) {
            cudaStreamDestroy(s_);
        }
    }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& o) noexcept : s_{std::exchange(o.s_, nullptr)} {}
    Stream& operator=(Stream&& o) noexcept {
        if (this != &o) {
            if (s_) cudaStreamDestroy(s_);
            s_ = std::exchange(o.s_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }

    void sync() {
        CUDA_CHECK(cudaStreamSynchronize(s_));
    }

private:
    cudaStream_t s_{nullptr};
};

}  // namespace cudalab
