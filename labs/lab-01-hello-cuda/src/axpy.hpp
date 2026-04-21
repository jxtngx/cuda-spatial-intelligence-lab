#pragma once

#include "stream.hpp"

#include <cstddef>
#include <span>
#include <type_traits>

namespace cudalab {

template <typename T>
concept GpuFloat = std::is_floating_point_v<T>;  // extend for __half / bf16 later

// y = alpha * x + y. Three implementations live in axpy.cu:
//   v0 = naive  (one thread per element, scalar load/store)
//   v1 = vec4   (one thread per 4 elements via float4 load/store)
//   v2 = stride (grid-stride loop on top of vec4)
enum class AxpyVersion { v0_naive, v1_vec4, v2_stride };

template <GpuFloat T>
void axpy(T alpha,
          std::span<const T> x,
          std::span<T>       y,
          Stream&            s,
          AxpyVersion        v = AxpyVersion::v2_stride);

// Explicit instantiations live in axpy.cu.

}  // namespace cudalab
