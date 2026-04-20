// Microbench for Week-03 reduction, scan, and histogram.
//
// Reports: best-of-K cudaEvent_t timing, achieved GB/s, and % of the
// CUB baseline. The perf rule for the week:
//   reduce v4 / cub::DeviceReduce  >= 80%
//   scan  best / cub::DeviceScan   >= 70%
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

#include "histogram.hpp"
#include "reduce.hpp"
#include "scan.hpp"

namespace {

float time_ms(auto&& fn, int iters = 50, int warmup = 10) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;
}

}  // namespace

int main() {
    constexpr std::size_t N = 1u << 28;  // 256M floats = 1 GiB

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    auto bw = [&](float ms) {
        return (static_cast<double>(N) * sizeof(float)) /
               (static_cast<double>(ms) * 1e-3) / 1e9;
    };

    float t_cub = time_ms([&] { week03::launch_reduce_cub(d_in, d_out, N, 0); });

    std::printf("=== Reduction (N=%zu, %.2f GiB) ===\n", N,
                static_cast<double>(N * sizeof(float)) / (1ull << 30));
    std::printf("%-28s %10s %10s %8s\n", "version", "ms", "GB/s", "%cub");
    std::printf("%-28s %10.3f %10.1f %7.1f%%\n", "cub::DeviceReduce",
                t_cub, bw(t_cub), 100.0);

    const char* names[] = {"v0 interleaved-divergent",
                           "v1 interleaved-strided",
                           "v2 sequential-addressing",
                           "v3 first-add-during-load",
                           "v4 warp-shuffle"};
    for (int v = 0; v < 5; ++v) {
        float t = time_ms([&] {
            week03::launch_reduce(d_in, d_out, N,
                                  static_cast<week03::ReduceVersion>(v), 0);
        });
        std::printf("%-28s %10.3f %10.1f %7.1f%%\n",
                    names[v], t, bw(t), 100.0 * t_cub / t);
    }

    cudaFree(d_in);
    cudaFree(d_out);

    // ---- Scan ----
    constexpr std::size_t S = 1024;
    float *d_si, *d_so;
    cudaMalloc(&d_si, S * sizeof(float));
    cudaMalloc(&d_so, S * sizeof(float));
    cudaMemset(d_si, 0, S * sizeof(float));

    float t_scan_cub = time_ms([&] { week03::launch_scan_cub(d_si, d_so, S, 0); });
    float t_hs = time_ms([&] {
        week03::launch_scan(d_si, d_so, S, week03::ScanVersion::HillisSteele, 0);
    });
    float t_cg = time_ms([&] {
        week03::launch_scan(d_si, d_so, S, week03::ScanVersion::CoopGroups, 0);
    });

    std::printf("\n=== Scan (N=%zu, single block) ===\n", S);
    std::printf("%-28s %10s %8s\n", "version", "ms", "%cub");
    std::printf("%-28s %10.4f %7.1f%%\n", "cub::DeviceScan", t_scan_cub, 100.0);
    std::printf("%-28s %10.4f %7.1f%%\n", "hillis-steele", t_hs,
                100.0 * t_scan_cub / t_hs);
    std::printf("%-28s %10.4f %7.1f%%\n", "cooperative-groups", t_cg,
                100.0 * t_scan_cub / t_cg);
    cudaFree(d_si); cudaFree(d_so);

    // ---- Histogram ----
    constexpr std::size_t H = 1u << 26;
    std::uint8_t* d_hi = nullptr;
    unsigned int* d_bins = nullptr;
    cudaMalloc(&d_hi, H);
    cudaMalloc(&d_bins, 256 * sizeof(unsigned int));
    cudaMemset(d_hi, 0, H);

    float t_hg = time_ms([&] {
        week03::launch_histogram(d_hi, d_bins, H,
                                 week03::HistVersion::GlobalAtomic, 0);
    });
    float t_hs2 = time_ms([&] {
        week03::launch_histogram(d_hi, d_bins, H,
                                 week03::HistVersion::SharedWarpAggregated, 0);
    });

    std::printf("\n=== Histogram (N=%zu uint8) ===\n", H);
    std::printf("%-28s %10s\n", "version", "ms");
    std::printf("%-28s %10.3f\n", "global-atomic", t_hg);
    std::printf("%-28s %10.3f\n", "shared+warp-aggregated", t_hs2);
    cudaFree(d_hi); cudaFree(d_bins);

    return 0;
}
