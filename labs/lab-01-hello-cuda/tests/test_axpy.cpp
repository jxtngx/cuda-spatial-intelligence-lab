#include "axpy.hpp"
#include "device_buffer.hpp"
#include "stream.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using cudalab::axpy;
using cudalab::AxpyVersion;
using cudalab::DeviceBuffer;
using cudalab::Stream;

namespace {

std::vector<float> make_random(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    std::generate(v.begin(), v.end(), [&] { return dist(rng); });
    return v;
}

float run_and_max_error(std::size_t n, AxpyVersion v) {
    auto h_x = make_random(n, /*seed=*/42);
    auto h_y = make_random(n, /*seed=*/43);

    std::vector<float> ref(n);
    constexpr float alpha = 0.5f;
    for (std::size_t i = 0; i < n; ++i) ref[i] = alpha * h_x[i] + h_y[i];

    Stream s;
    DeviceBuffer<float> d_x(n, s);
    DeviceBuffer<float> d_y(n, s);
    d_x.copy_from_host(h_x, s);
    d_y.copy_from_host(h_y, s);

    axpy<float>(alpha, d_x.span(), d_y.span(), s, v);

    std::vector<float> out(n);
    d_y.copy_to_host(out, s);
    s.sync();

    float max_err = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(out[i] - ref[i]));
    }
    return max_err;
}

}  // namespace

class AxpyAtSize : public ::testing::TestWithParam<std::size_t> {};

TEST_P(AxpyAtSize, Naive)  { EXPECT_LT(run_and_max_error(GetParam(), AxpyVersion::v0_naive),  1e-5f); }
TEST_P(AxpyAtSize, Vec4)   { EXPECT_LT(run_and_max_error(GetParam(), AxpyVersion::v1_vec4),   1e-5f); }
TEST_P(AxpyAtSize, Stride) { EXPECT_LT(run_and_max_error(GetParam(), AxpyVersion::v2_stride), 1e-5f); }

INSTANTIATE_TEST_SUITE_P(
    Sizes, AxpyAtSize,
    ::testing::Values(std::size_t{1}, std::size_t{17}, std::size_t{1u << 16},
                      std::size_t{1u << 28}));

TEST(AxpyEdge, ZeroSizeIsNoop) {
    Stream s;
    DeviceBuffer<float> d_x(0, s);
    DeviceBuffer<float> d_y(0, s);
    EXPECT_NO_THROW(axpy<float>(1.0f, d_x.span(), d_y.span(), s));
    s.sync();
}
