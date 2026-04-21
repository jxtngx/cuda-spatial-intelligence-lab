// Single-block inclusive scan, two flavors.
//
// Tier A scaffold: both kernels compile and round-trip a float[1024]
// correctly. Two TODO(student): markers below mark (1) the
// Hillis-Steele inner step, and (2) which cooperative_groups partition
// the cg version uses (whole block vs. tile<32>).
#include "scan.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

namespace week03 {

namespace cg = cooperative_groups;

static constexpr int SCAN_BLOCK = 1024;

__global__ void scan_hillis_steele(const float* __restrict__ in,
                                   float* __restrict__ out, std::size_t n) {
    // Two-buffer ping-pong (PMPP 4e §11.2, Fig 11.4).
    __shared__ float buf[2][SCAN_BLOCK];
    unsigned tid = threadIdx.x;
    int pin = 0, pout = 1;

    buf[pin][tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();

    // TODO(student): Hillis-Steele step.
    // For each `offset` in {1, 2, 4, ..., n/2}:
    //   buf[pout][tid] = buf[pin][tid] +
    //                    (tid >= offset ? buf[pin][tid - offset] : 0.0f);
    //   __syncthreads();
    //   swap(pin, pout);
    // Total work O(n log n); span O(log n). Inferior to Brent-Kung in
    // work, equal in span — Brent-Kung is the Week-04 stretch.
    for (unsigned offset = 1; offset < n; offset <<= 1) {
        float v = buf[pin][tid];
        if (tid >= offset) v += buf[pin][tid - offset];
        buf[pout][tid] = v;
        __syncthreads();
        int tmp = pin; pin = pout; pout = tmp;
    }

    if (tid < n) out[tid] = buf[pin][tid];
}

__global__ void scan_coop_groups(const float* __restrict__ in,
                                 float* __restrict__ out, std::size_t n) {
    unsigned tid = threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;

    // TODO(student): pick the cooperative_groups partition.
    // Option A (recommended): per-warp tile<32> scan, then a one-warp
    // reduction of the per-warp totals, then add the offset back.
    // Option B (simpler, slower): cg::inclusive_scan on the whole
    // thread_block via cg::this_thread_block(). Try both, report the
    // delta in §7 Results.
    auto block = cg::this_thread_block();
    v = cg::inclusive_scan(block, v, cg::plus<float>{});

    if (tid < n) out[tid] = v;
}

void launch_scan(const float* d_in, float* d_out, std::size_t n,
                 ScanVersion version, cudaStream_t stream) {
    if (n == 0) return;
    // Tier-A constraint: single-block. Asserted by the test harness.
    int block = SCAN_BLOCK;
    switch (version) {
        case ScanVersion::HillisSteele:
            scan_hillis_steele<<<1, block, 0, stream>>>(d_in, d_out, n); break;
        case ScanVersion::CoopGroups:
            scan_coop_groups<<<1, block, 0, stream>>>(d_in, d_out, n); break;
    }
}

}  // namespace week03
