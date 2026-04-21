#pragma once

#include "cuda_check.hpp"
#include "stream.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace cudalab {

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// Move-only RAII owner of a device-side allocation made via cudaMallocAsync.
// On destruction, frees on the default stream (null) — explicit lifetime
// management with a specific stream is the user's responsibility for hot paths.
template <TriviallyCopyable T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n, Stream& s) : n_{n} {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&p_),
                                   n * sizeof(T), s.get()));
    }

    ~DeviceBuffer() {
        if (p_) {
            cudaFreeAsync(p_, /*default stream*/ nullptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : p_{std::exchange(o.p_, nullptr)}, n_{std::exchange(o.n_, 0)} {}

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (p_) cudaFreeAsync(p_, nullptr);
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }

    [[nodiscard]] T*           data() noexcept       { return p_; }
    [[nodiscard]] const T*     data() const noexcept { return p_; }
    [[nodiscard]] std::size_t  size() const noexcept { return n_; }
    [[nodiscard]] std::span<T>       span() noexcept       { return {p_, n_}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {p_, n_}; }

    void copy_from_host(std::span<const T> host, Stream& s) {
        CUDA_CHECK(cudaMemcpyAsync(p_, host.data(), n_ * sizeof(T),
                                   cudaMemcpyHostToDevice, s.get()));
    }

    void copy_to_host(std::span<T> host, Stream& s) const {
        CUDA_CHECK(cudaMemcpyAsync(host.data(), p_, n_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, s.get()));
    }

private:
    T*          p_{nullptr};
    std::size_t n_{0};
};

}  // namespace cudalab
