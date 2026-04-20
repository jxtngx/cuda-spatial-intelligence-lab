#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "histogram.hpp"
#include "reduce.hpp"
#include "scan.hpp"

namespace {

constexpr float kAbsTol = 1e-2f;  // single-precision sum tolerance
constexpr float kRelTol = 1e-4f;

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t bytes = 0;
    explicit DeviceArray(std::size_t n) : bytes(n * sizeof(T)) {
        cudaMalloc(&ptr, bytes);
    }
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

}  // namespace

class ReduceTest : public ::testing::TestWithParam<week03::ReduceVersion> {};

TEST_P(ReduceTest, MatchesCpuReference) {
    constexpr std::size_t N = 1u << 20;  // 1M floats
    std::vector<float> h(N);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : h) v = d(rng);
    double cpu = std::accumulate(h.begin(), h.end(), 0.0);

    DeviceArray<float> in(N);
    DeviceArray<float> out(1);
    cudaMemcpy(in.ptr, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    week03::launch_reduce(in.ptr, out.ptr, N, GetParam(), /*stream*/ 0);
    cudaDeviceSynchronize();

    float gpu = 0.0f;
    cudaMemcpy(&gpu, out.ptr, sizeof(float), cudaMemcpyDeviceToHost);

    float tol = std::max(kAbsTol, kRelTol * static_cast<float>(std::abs(cpu)));
    EXPECT_NEAR(gpu, static_cast<float>(cpu), tol)
        << "version=" << static_cast<int>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    AllVersions, ReduceTest,
    ::testing::Values(week03::ReduceVersion::V0_InterleavedDivergent,
                      week03::ReduceVersion::V1_InterleavedStrided,
                      week03::ReduceVersion::V2_SequentialAddressing,
                      week03::ReduceVersion::V3_FirstAddDuringLoad,
                      week03::ReduceVersion::V4_WarpShuffle));

class ScanTest : public ::testing::TestWithParam<week03::ScanVersion> {};

TEST_P(ScanTest, MatchesCpuReference) {
    constexpr std::size_t N = 1024;
    std::vector<float> h(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : h) v = d(rng);

    std::vector<float> ref(N);
    float acc = 0.0f;
    for (std::size_t i = 0; i < N; ++i) { acc += h[i]; ref[i] = acc; }

    DeviceArray<float> in(N);
    DeviceArray<float> out(N);
    cudaMemcpy(in.ptr, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    week03::launch_scan(in.ptr, out.ptr, N, GetParam(), 0);
    cudaDeviceSynchronize();

    std::vector<float> gpu(N);
    cudaMemcpy(gpu.data(), out.ptr, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(gpu[i] - ref[i]));
    EXPECT_LT(max_err, 1e-2f)
        << "version=" << static_cast<int>(GetParam())
        << " max_err=" << max_err;
}

INSTANTIATE_TEST_SUITE_P(AllVersions, ScanTest,
                         ::testing::Values(week03::ScanVersion::HillisSteele,
                                           week03::ScanVersion::CoopGroups));

TEST(Histogram, MatchesCpuReference) {
    constexpr std::size_t N = 1u << 20;
    std::vector<std::uint8_t> h(N);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& v : h) v = static_cast<std::uint8_t>(d(rng));
    std::vector<unsigned int> ref(256, 0);
    for (auto v : h) ++ref[v];

    DeviceArray<std::uint8_t> in(N);
    DeviceArray<unsigned int> bins(256);
    cudaMemcpy(in.ptr, h.data(), N, cudaMemcpyHostToDevice);

    for (auto version : {week03::HistVersion::GlobalAtomic,
                         week03::HistVersion::SharedWarpAggregated}) {
        week03::launch_histogram(in.ptr, bins.ptr, N, version, 0);
        cudaDeviceSynchronize();

        std::vector<unsigned int> gpu(256);
        cudaMemcpy(gpu.data(), bins.ptr, 256 * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        for (int b = 0; b < 256; ++b)
            ASSERT_EQ(gpu[b], ref[b])
                << "bin " << b << " version=" << static_cast<int>(version);
    }
}
